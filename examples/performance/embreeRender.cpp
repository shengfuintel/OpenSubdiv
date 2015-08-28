#include "embree2/rtcore.h"
#include "embree2/rtcore_scene.h"
#include "embree2/rtcore_geometry.h"
#include "embree2/rtcore_ray.h"

#include <tbb/parallel_for.h>

#include <string.h>
#include <math.h>
#include <float.h>
#include <string>
#include <stdio.h>

RTCScene g_scene;

unsigned g_geomID;

static const unsigned int *g_indices;
static const float        *g_normals;
static float g_lightDir [] = {0.0, 1.0, -1.0};
static float g_ambient = 0.3f;
static bool  g_shadow = false;

void normalize(float *n) {
    float m = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    n[0] /= m;
    n[1] /= m;
    n[2] /= m;
}

float dot(float v1[3], float v2[3])
{ 
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

void initScene(const float *vertexBuffer, const unsigned int *indexBuffer,
               int nTriangle, int nVertex) {
    if(!g_scene) {
        std::string cfg;
//        std::string cfg ="threads=16, accel=bvh8.triangle8";
        rtcInit(cfg.c_str());
    }
    
    if(g_scene) 
        rtcDeleteScene(g_scene);
    
    g_scene = rtcNewScene( RTC_SCENE_DYNAMIC | RTC_SCENE_COHERENT | RTC_SCENE_HIGH_QUALITY | RTC_SCENE_ROBUST, 
                           RTC_INTERSECT1 | RTC_INTERSECT8);
    g_geomID = rtcNewTriangleMesh (g_scene, RTC_GEOMETRY_STATIC, nTriangle, nVertex);
    
    rtcSetBuffer(g_scene, g_geomID, RTC_INDEX_BUFFER, (void*)indexBuffer, 0, 3*sizeof(uint));
    rtcSetBuffer(g_scene, g_geomID, RTC_VERTEX_BUFFER0, (void*)vertexBuffer, 0, 3*sizeof(float));       
    
    normalize(g_lightDir); 
}

void updateScene(const float *vertexBuffer, const unsigned int *indexBuffer, 
                 const float *normalBuffer,int nTriangle, int nVertex) {      
    rtcUpdate(g_scene, g_geomID);
      
    rtcCommit(g_scene);    
    
    g_indices = indexBuffer;
    g_normals = normalBuffer;
}

unsigned int renderPixel(float x, float y, const float * vx, const float * vy, const float * vz, const float * p)
{
  /* initialize ray */
  RTCRay ray;
  ray.org[0] = p[0];
  ray.org[1] = p[1];
  ray.org[2] = p[2];    
  ray.dir[0] = x * vx[0] + y * vy[0] - vz[0];
  ray.dir[1] = x * vx[1] + y * vy[1] - vz[1];
  ray.dir[2] = x * vx[2] + y * vy[2] - vz[2];      
  normalize(ray.dir);
  ray.tnear = 0.0f;
  ray.tfar = FLT_MAX;
  ray.geomID = (int)RTC_INVALID_GEOMETRY_ID;
  ray.primID = (int)RTC_INVALID_GEOMETRY_ID;
  ray.mask = -1;
  ray.time = 0;

  /* intersect ray with scene */
  rtcIntersect(g_scene,ray);

  /* shade pixel */
  if (ray.geomID == RTC_INVALID_GEOMETRY_ID) return 55;
  
  int triangleId = ray.primID;
  int id1 = g_indices[triangleId * 3    ];
  int id2 = g_indices[triangleId * 3 + 1];
  int id3 = g_indices[triangleId * 3 + 2];    
      
  const float *n1 = g_normals + 3 * id1;        
  const float *n2 = g_normals + 3 * id2;
  const float *n3 = g_normals + 3 * id3;  
      
  float n[3];
  float u = ray.u;
  float v = ray.v;    
  float w = 1.0f - u - v;          
      
  for(int i=0; i<3; i++) {
      n[i]    =    n1[i] * w  +   n2[i] * u +    n3[i] * v;       
  }             
  normalize(n);
      
  float g = fabs(dot(ray.dir, g_lightDir));

/*
  float intensity = 0;
  float hitPos[3];
  for(int i=0; i<3; i++)
      hitPos[i] = ray.org[i] + ray.tfar * ray.dir[i];

#define AMBIENT_OCCLUSION_SAMPLES 64
  // trace some ambient occlusion rays 
  int seed = 34*x+12*y;
  for (int i=0; i<AMBIENT_OCCLUSION_SAMPLES; i++) 
  {
      RTCRay shadow;
            
      const float oneOver10000f = 1.f/10000.f;
      seed = 1103515245 * seed + 12345;
      shadow.dir[0] = (seed%10000)*oneOver10000f;
      seed = 1103515245 * seed + 12345;
      shadow.dir[1] = (seed%10000)*oneOver10000f;
      seed = 1103515245 * seed + 12345;
      shadow.dir[2] = (seed%10000)*oneOver10000f;
    
      // initialize shadow ray 

      shadow.org[0] = hitPos[0];
      shadow.org[1] = hitPos[1];      
      shadow.org[2] = hitPos[2];      
      shadow.tnear = 0.1f;
      shadow.tfar = FLT_MAX;
      shadow.geomID = (int)RTC_INVALID_GEOMETRY_ID;
      shadow.primID = (int)RTC_INVALID_GEOMETRY_ID;
      shadow.mask = -1;
      shadow.time = 0;
    
      // trace shadow ray 
      rtcOccluded(g_scene,shadow);
    
      // add light contribution 
      if (shadow.geomID == RTC_INVALID_GEOMETRY_ID)
        intensity += 1.0f;   
  }
  intensity *= 1.0f/AMBIENT_OCCLUSION_SAMPLES;

  // shade pixel 
  g =  intensity; 
  */ 
  unsigned char c = g * 255;
  return c << 16 | c << 8 | c;
}

void renderPixel8(void *allOnMask, float *x, float *y, unsigned int *color, 
                  const float * vx, const float * vy, const float * vz, const float * p, bool shadowOn)
{
    /* initialize ray */
    RTCRay8 ray;
    for(int i=0; i<8; i++) {
        ray.orgx[i] = p[0];
        ray.orgy[i] = p[1];
        ray.orgz[i] = p[2];            
      
        float dir[3];
        dir[0] = x[i] * vx[0] + y[i] * vy[0] - vz[0];
        dir[1] = x[i] * vx[1] + y[i] * vy[1] - vz[1];
        dir[2] = x[i] * vx[2] + y[i] * vy[2] - vz[2];      
        normalize(dir);
      
        ray.dirx[i] = dir[0];
        ray.diry[i] = dir[1];
        ray.dirz[i] = dir[2];            
      
        ray.tnear[i] = 0.0f;
        ray.tfar[i] = FLT_MAX;
        ray.geomID[i] = (int)RTC_INVALID_GEOMETRY_ID;
        ray.primID[i] = (int)RTC_INVALID_GEOMETRY_ID;
        ray.mask[i] = -1;
        ray.time[i] = 0;
    }

    /* intersect ray with scene */
    rtcIntersect8(allOnMask, g_scene,ray);

    for(int i=0; i<8; i++) {
        /* shade pixel */
        if (ray.geomID[i] == RTC_INVALID_GEOMETRY_ID) {
            color[i] = 55;
            continue;
        }
  
        if(shadowOn) {
            float hitPos[3];
            hitPos[0] = ray.orgx[i] + ray.tfar[i] * ray.dirx[i];
            hitPos[1] = ray.orgy[i] + ray.tfar[i] * ray.diry[i];
            hitPos[2] = ray.orgz[i] + ray.tfar[i] * ray.dirz[i];                
            
            RTCRay shadow;
            shadow.org[0] = hitPos[0];
            shadow.org[1] = hitPos[1];      
            shadow.org[2] = hitPos[2];   
            shadow.dir[0] = -g_lightDir[0];   
            shadow.dir[1] = -g_lightDir[1];
            shadow.dir[2] = -g_lightDir[2];                
            shadow.tnear = 0.001f;
            shadow.tfar = FLT_MAX;
            shadow.geomID = (int)RTC_INVALID_GEOMETRY_ID;
            shadow.primID = (int)RTC_INVALID_GEOMETRY_ID;
            shadow.mask = -1;
            shadow.time = 0;
    
            // trace shadow ray 
            rtcOccluded(g_scene,shadow);
            
            if (shadow.geomID != RTC_INVALID_GEOMETRY_ID) {
                color[i] = g_ambient;
                continue;
            }            
        }
        
        int triangleId = ray.primID[i];
        int id1 = g_indices[triangleId * 3    ];
        int id2 = g_indices[triangleId * 3 + 1];
        int id3 = g_indices[triangleId * 3 + 2];    
      
        const float *n1 = g_normals + 3 * id1;        
        const float *n2 = g_normals + 3 * id2;
        const float *n3 = g_normals + 3 * id3;  
      
        float n[3];
        float u = ray.u[i];
        float v = ray.v[i];    
        float w = 1.0f - u - v;          
      
        for(int j=0; j<3; j++) {
            n[j] = n1[j] * w  + n2[j] * u + n3[j] * v;       
        }             
        normalize(n);
                      
        float g = fabs(dot(g_lightDir, n));

        unsigned char c = g * 255;
        color[i] = c << 16 | c << 8 | c;
    }
}

#define BUNDLE

#ifdef BUNDLE 

void renderScene(unsigned int *buffer, int width, int height,                
                 float *eye, float *xaxis, float *yaxis, float *zaxis,
                 float fov,  float znear, bool shadowOn)
{
    float aspect = (float)width / (float) height;
    
    float ysize = znear * tan(fov * M_PI / 180.0f / 2.0f);
    float xsize = ysize * aspect; 
    
    float dx = 2.0f * xsize / width;
    float dy = 2.0f * ysize / height;
    
    float sx = -xsize;
    float sy = -ysize;
    
    float z[3];
    z[0] = znear * zaxis[0];
    z[1] = znear * zaxis[1];
    z[2] = znear * zaxis[2];        
       
    int xBlock = width  / 16;
    int yBlock = height / 16;
       
     __declspec( align(64) ) unsigned int allOnMask[16];
     memset(&allOnMask[0], 0xff, 16 * sizeof(unsigned int));
     
    tbb::blocked_range<int> range = tbb::blocked_range<int>(0, xBlock * yBlock);
    tbb::parallel_for(range, [&](const tbb::blocked_range<int> &r)
    {    
        for(int i=r.begin(); i<r.end(); i++) {                
        //for(int i=0; i<xBlock * yBlock; i++) 
            int yi = i / xBlock;
            int xi = i - yi * xBlock;
            
            int xstart, ystart, xend, yend;
            xstart = xi * 16;
            xend   = xstart + 16;
            ystart = yi * 16;
            yend   = ystart + 16;
                  
            for(int m=ystart; m<yend; m++) 
            {
                for(int n=xstart; n<xend; n+=8) 
                {
                    float x[8], y[8];
                    unsigned int color[8];
                    for(int k=0; k<8; k++) {
                        x[k] = sx + dx * (n+k);
                        y[k] = sy + dy * m;
                    }
            
                    renderPixel8((void *)&allOnMask[0], x, y, color, xaxis, yaxis, z, eye, shadowOn);
                    
                    for(int k=0; k<8; k++) 
                        buffer[width * m + n + k] = color[k];
                }
            }
        }
    });
    
}                 

#else

void renderScene(unsigned int *buffer, int width, int height,
                 float *eye, float *xaxis, float *yaxis, float *zaxis,
                 float fov,  float znear, bool shadowOn)
{
    float aspect = (float)width / (float) height;
    
    float ysize = znear * tan(fov * M_PI / 180.0f / 2.0f);
    float xsize = ysize * aspect; 
    
    float dx = 2.0f * xsize / width;
    float dy = 2.0f * ysize / height;
    
    float sx = -xsize;
    float sy = -ysize;
    
    float z[3];
    z[0] = znear * zaxis[0];
    z[1] = znear * zaxis[1];
    z[2] = znear * zaxis[2];        
       
    int xBlock = width  / 16;
    int yBlock = height / 16;
    tbb::blocked_range<int> range = tbb::blocked_range<int>(0, xBlock * yBlock);
    tbb::parallel_for(range, [&](const tbb::blocked_range<int> &r)
    {    
        for(int i=r.begin(); i<r.end(); i++) {                
        //for(int i=0; i<xBlock * yBlock; i++) 
            int yi = i / xBlock;
            int xi = i - yi * xBlock;
            
            int xstart, ystart, xend, yend;
            xstart = xi * 16;
            xend   = xstart + 16;
            ystart = yi * 16;
            yend   = ystart + 16;
                  
            for(int m=ystart; m<yend; m++) 
            {
                for(int n=xstart; n<xend; n++) 
                {
                    float x = sx + dx * n;
                    float y = sy + dy * m;
            
                    buffer[width * m + n] = renderPixel(x, y, xaxis, yaxis, z, eye);
                }
            }
        }
    });
    
}                 

#endif
