//
//   Copyright 2013 Pixar
//
//   Licensed under the Apache License, Version 2.0 (the "Apache License")
//   with the following modification; you may not use this file except in
//   compliance with the Apache License and the following modification to it:
//   Section 6. Trademarks. is deleted and replaced with:
//
//   6. Trademarks. This License does not grant permission to use the trade
//      names, trademarks, service marks, or product names of the Licensor
//      and its affiliates, except as required to comply with Section 4(c) of
//      the License and to reproduce the content of the NOTICE file.
//
//   You may obtain a copy of the Apache License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the Apache License with the above modification is
//   distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
//   KIND, either express or implied. See the Apache License for the specific
//   language governing permissions and limitations under the Apache License.
//

#include "../common/glUtils.h"

#include <GLFW/glfw3.h>
GLFWwindow* g_window=0;
GLFWmonitor* g_primary=0;

#include <far/error.h>

#include <osd/cpuEvaluator.h>
#include <osd/ispcEvaluator.h>
#include <osd/cpuVertexBuffer.h>
#include <osd/cpuPatchTable.h>
#include <osd/mesh.h>

using namespace OpenSubdiv;

#include "../../regression/common/far_utils.h"
#include "../common/glHud.h"
#include "../common/glUtils.h"
#include "../common/glControlMeshDisplay.h"
#include "../common/objAnim.h"
#include "../common/simple_math.h"
#include "../common/stopwatch.h"

#include <tbb/parallel_for.h>

#include <cfloat>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

void initScene(const float *vertexBuffer, const unsigned int *indexBuffer,  int nTriangle, int nVertex);

void updateScene(const float *vertexBuffer, const unsigned int *indexBuffer, const float *normalBuffer, 
                 int nTriangle, int nVertex);

enum KernelType { kCPU = 0,
                  kOPENMP = 1,
                  kTBB = 2,
                  kCUDA = 3,
                  kCL = 4,
                  kGLSL = 5,
                  kGLSLCompute = 6 };

enum DisplayStyle { kDisplayStyleWire,
                    kDisplayStyleShaded,
                    kDisplayStyleWireOnShaded };

enum ShadingMode { kShadingMaterial,
                   kShadingVaryingColor,
                   kShadingInterleavedVaryingColor,
                   kShadingFaceVaryingColor,
                   kShadingPatchType,
                   kShadingPatchCoord,
                   kShadingNormal };

enum EndCap      { kEndCapNone = 0,
                   kEndCapBSplineBasis,
                   kEndCapGregoryBasis,
                   kEndCapLegacyGregory };

enum HudCheckBox { kHUD_CB_ANIMATE_VERTICES,
                   kHUD_CB_FREEZE,
                   kHUD_CB_SHADOW };

int g_currentShape = 0;

ObjAnim const * g_objAnim = 0;

bool g_axis=true;

int   g_frame = 0,
      g_repeatCount = 0;
float g_animTime = 0;

// GUI variables
int   g_fullscreen = 0,
      g_freeze = 0,
      g_shadow = 0,
      g_endCap = kEndCapBSplineBasis,
      g_singleCreasePatch = 1,
      g_mbutton[3] = {0, 0, 0},
      g_running = 1;

float g_rotate[2] = {-36, 17},
      g_dolly = 5,
      g_pan[2] = {0, 0},
      g_center[3] = {0, 0, 0},
      g_size = 0;

int   g_prev_x = 0,
      g_prev_y = 0;

int   g_width = 800,
      g_height = 800;

unsigned int *g_frameBuffer = NULL;

GLhud g_hud;
GLControlMeshDisplay g_controlMeshDisplay;

// performance
float g_cpuTime = 0;
float g_gpuTime = 0;
float g_computeTime = 0;
float g_evalTime = 0;

Stopwatch g_fpsTimer;

// geometry
std::vector<float> g_orgPositions;

// input and output vertex data
class EvalOutputBase {
public:
    virtual ~EvalOutputBase() {}
    virtual const float *GetVertexData() const = 0;
    virtual const float *GetDerivatives() const = 0;
    virtual void UpdateData(const float *src, int startVertex, int numVertices) = 0;
    virtual void UpdateVaryingData(const float *src, int startVertex, int numVertices) = 0;
    virtual void Refine() = 0;
    virtual void EvalPatches() = 0;
    virtual void EvalPatchesWithDerivatives() = 0;
    virtual void EvalPatchesVarying() = 0;
    virtual void UpdatePatchCoords(
        std::vector<Osd::PatchCoord> const &patchCoords) = 0;
};

// note: Since we don't have a class for device-patchcoord container in osd,
// we cheat to use vertexbuffer as a patch-coord (5int) container.
//
// Please don't follow the pattern in your actual application.
//
template<typename SRC_VERTEX_BUFFER, typename EVAL_VERTEX_BUFFER,
         typename STENCIL_TABLE, typename PATCH_TABLE, typename EVALUATOR,
         typename DEVICE_CONTEXT = void>
class EvalOutput : public EvalOutputBase {
public:
    typedef Osd::EvaluatorCacheT<EVALUATOR> EvaluatorCache;

    EvalOutput(Far::StencilTable const *vertexStencils,
               Far::StencilTable const *varyingStencils,
               int numCoarseVerts, int numTotalVerts, int numParticles,
               Far::PatchTable const *patchTable,
               EvaluatorCache *evaluatorCache = NULL,
               DEVICE_CONTEXT *deviceContext = NULL)
        : _srcDesc(       /*offset*/ 0, /*length*/ 3, /*stride*/ 3),
          _srcVaryingDesc(/*offset*/ 0, /*length*/ 3, /*stride*/ 3),
          _vertexDesc(    /*offset*/ 0, /*legnth*/ 3, /*stride*/ 3),
          _varyingDesc(   /*offset*/ 3, /*legnth*/ 3, /*stride*/ 3),
          _duDesc(        /*offset*/ 0, /*legnth*/ 3, /*stride*/ 6),
          _dvDesc(        /*offset*/ 3, /*legnth*/ 3, /*stride*/ 6),
          _deviceContext(deviceContext) {
        _srcData = SRC_VERTEX_BUFFER::Create(3, numTotalVerts, _deviceContext);
        _srcVaryingData = SRC_VERTEX_BUFFER::Create(3, numTotalVerts, _deviceContext);
        _vertexData = EVAL_VERTEX_BUFFER::Create(6, numParticles, _deviceContext);
        _derivatives = EVAL_VERTEX_BUFFER::Create(6, numParticles, _deviceContext);
        _patchTable = PATCH_TABLE::Create(patchTable, _deviceContext);
        _patchCoords = NULL;
        _numCoarseVerts = numCoarseVerts;
        _vertexStencils =
            Osd::convertToCompatibleStencilTable<STENCIL_TABLE>(vertexStencils, _deviceContext);
        _varyingStencils =
            Osd::convertToCompatibleStencilTable<STENCIL_TABLE>(varyingStencils, _deviceContext);
        _evaluatorCache = evaluatorCache;
    }
    ~EvalOutput() {
        delete _srcData;
        delete _srcVaryingData;
        delete _vertexData;
        delete _derivatives;
        delete _patchTable;
        delete _patchCoords;
        delete _vertexStencils;
        delete _varyingStencils;
    }

    virtual const float *GetVertexData() const {
        return _vertexData->BindCpuBuffer();
    }
    
    virtual const float *GetDerivatives() const {
        return _derivatives->BindCpuBuffer();
    }

    virtual void UpdateData(const float *src, int startVertex, int numVertices) {
        _srcData->UpdateData(src, startVertex, numVertices, _deviceContext);
    }
    virtual void UpdateVaryingData(const float *src, int startVertex, int numVertices) {
        _srcVaryingData->UpdateData(src, startVertex, numVertices, _deviceContext);
    }
    virtual void Refine() {
        Osd::BufferDescriptor dstDesc = _srcDesc;
        dstDesc.offset += _numCoarseVerts * _srcDesc.stride;

        EVALUATOR const *evalInstance = OpenSubdiv::Osd::GetEvaluator<EVALUATOR>(
            _evaluatorCache, _srcDesc, dstDesc, _deviceContext);

        EVALUATOR::EvalStencils(_srcData, _srcDesc,
                                _srcData, dstDesc,
                                _vertexStencils,
                                evalInstance,
                                _deviceContext);
        if(_varyingStencils) {
            dstDesc = _srcVaryingDesc;
            dstDesc.offset += _numCoarseVerts * _srcVaryingDesc.stride;
            evalInstance = OpenSubdiv::Osd::GetEvaluator<EVALUATOR>(
                _evaluatorCache, _srcVaryingDesc, dstDesc, _deviceContext);

            EVALUATOR::EvalStencils(_srcVaryingData, _srcVaryingDesc,
                                    _srcVaryingData, dstDesc,
                                    _varyingStencils,
                                    evalInstance,
                                    _deviceContext);
        }
    }
    virtual void EvalPatches() {
        EVALUATOR const *evalInstance = OpenSubdiv::Osd::GetEvaluator<EVALUATOR>(
            _evaluatorCache, _srcDesc, _vertexDesc, _deviceContext);

        EVALUATOR::EvalPatches(
            _srcData, _srcDesc,
            _vertexData, _vertexDesc,
            _patchCoords->GetNumVertices(),
            _patchCoords,
            _patchTable, evalInstance, _deviceContext);
    }
    virtual void EvalPatchesWithDerivatives() {
        EVALUATOR const *evalInstance = OpenSubdiv::Osd::GetEvaluator<EVALUATOR>(
            _evaluatorCache, _srcDesc, _vertexDesc, _duDesc, _dvDesc, _deviceContext);
        EVALUATOR::EvalPatches(
            _srcData, _srcDesc,
            _vertexData, _vertexDesc,
            _derivatives, _duDesc,
            _derivatives, _dvDesc,
            _patchCoords->GetNumVertices(),
            _patchCoords,
            _patchTable, evalInstance, _deviceContext);
    }
    virtual void EvalPatchesVarying() {
        EVALUATOR const *evalInstance = OpenSubdiv::Osd::GetEvaluator<EVALUATOR>(
            _evaluatorCache, _srcVaryingDesc, _varyingDesc, _deviceContext);

        EVALUATOR::EvalPatches(
            _srcVaryingData, _srcVaryingDesc,
            // varyingdata is interleved in vertexData.
            _vertexData, _varyingDesc,
            _patchCoords->GetNumVertices(),
            _patchCoords,
            _patchTable, evalInstance, _deviceContext);
    }
    virtual void UpdatePatchCoords(
        std::vector<Osd::PatchCoord> const &patchCoords) {
        if (_patchCoords and
            _patchCoords->GetNumVertices() != (int)patchCoords.size()) {
            delete _patchCoords;
            _patchCoords = NULL;
        }
        if (not _patchCoords) {
            _patchCoords = EVAL_VERTEX_BUFFER::Create(5,
                                                      (int)patchCoords.size(),
                                                      _deviceContext);
        }
        _patchCoords->UpdateData((float*)&patchCoords[0], 0, (int)patchCoords.size(), _deviceContext);
    }
private:
    SRC_VERTEX_BUFFER *_srcData;
    SRC_VERTEX_BUFFER *_srcVaryingData;
    EVAL_VERTEX_BUFFER *_vertexData;
    EVAL_VERTEX_BUFFER *_derivatives;
    EVAL_VERTEX_BUFFER *_varyingData;
    EVAL_VERTEX_BUFFER *_patchCoords;
    PATCH_TABLE *_patchTable;
    Osd::BufferDescriptor _srcDesc;
    Osd::BufferDescriptor _srcVaryingDesc;
    Osd::BufferDescriptor _vertexDesc;
    Osd::BufferDescriptor _varyingDesc;
    Osd::BufferDescriptor _duDesc;
    Osd::BufferDescriptor _dvDesc;
    int _numCoarseVerts;

    STENCIL_TABLE const *_vertexStencils;
    STENCIL_TABLE const *_varyingStencils;

    EvaluatorCache *_evaluatorCache;
    DEVICE_CONTEXT *_deviceContext;
};

Far::PatchTable const *g_patchTable = NULL;
EvalOutputBase        *g_evalOutput = NULL;

// tessellated mesh
std::vector<unsigned int>    g_triangleIndexBuffer;
std::vector<float>           g_normalBuffer;
std::vector<Osd::PatchCoord> g_patchCoordBuffer;
int g_nSamplePoints = 0;
int g_totalPatches = 0;

int g_level = 2;
int g_tessLevel = 1;
int g_tessLevelMin = 1;
float g_moveScale = 0.0f;

struct Transform {
    float ModelViewMatrix[16];
    float ProjectionMatrix[16];
    float ModelViewProjectionMatrix[16];
    float ModelViewInverseMatrix[16];
} g_transformData;

//------------------------------------------------------------------------------

#include "init_shapes.h"

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
static void
updateGeom() {
    std::vector<float> vertex;

    int nverts = 0;
    int stride = 3;

    nverts = (int)g_orgPositions.size() / 3;
    vertex.reserve(nverts*stride);

    const float *p = &g_orgPositions[0];
    float r = sin(g_frame*0.001f) * g_moveScale;
    for (int i = 0; i < nverts; ++i) {
        float ct = cos(p[2] * r);
        float st = sin(p[2] * r);
        vertex.push_back( p[0]*ct + p[1]*st);
        vertex.push_back(-p[0]*st + p[1]*ct);
        vertex.push_back( p[2]);
        p += 3;
    }

    // Run Compute pass to pose the control vertices ---------------------------
    Stopwatch s;
    s.Start();

    // update coarse vertices
    g_evalOutput->UpdateData(&vertex[0], 0, nverts);

    // Refine
    g_evalOutput->Refine();

    s.Stop();
    g_computeTime = float(s.GetElapsed() * 1000.0f);


    // Run Eval pass to get the samples locations ------------------------------
    s.Start();

    // update patchcoord to be evaluated
    g_evalOutput->UpdatePatchCoords(g_patchCoordBuffer);

    // evaluate positions
    g_evalOutput->EvalPatchesWithDerivatives();
    
    // compute normal
    const float *derivatives = g_evalOutput->GetDerivatives();
    tbb::blocked_range<int> range = tbb::blocked_range<int>(0, g_nSamplePoints);
    tbb::parallel_for(range, [&](const tbb::blocked_range<int> &r)
    { 
        for(int i=r.begin(); i<r.end(); i++) { 
        //for(int i=0; i<g_nSamplePoints; i++) {
            const float *du = derivatives + i * 6;
            const float *dv = derivatives + i * 6 + 3;        
        
            float *n = &g_normalBuffer[0] + i * 3;
            cross(n, du, dv);
            normalize(n);
        }
    });
    
    s.Stop();

    g_evalTime = float(s.GetElapsed());
    
    const float *pVertex = g_evalOutput->GetVertexData();
    updateScene(pVertex, &g_triangleIndexBuffer[0], &g_normalBuffer[0], 
                g_triangleIndexBuffer.size()/3, g_nSamplePoints);
}

//------------------------------------------------------------------------------
static const char *
getKernelName(int kernel) {

         if (kernel == kCPU)
        return "CPU";
    else if (kernel == kOPENMP)
        return "OpenMP";
    else if (kernel == kTBB)
        return "TBB";
    else if (kernel == kCUDA)
        return "Cuda";
    else if (kernel == kGLSL)
        return "GLSL TransformFeedback";
    else if (kernel == kGLSLCompute)
        return "GLSL Compute";
    else if (kernel == kCL)
        return "OpenCL";
    return "Unknown";
}

int setupPatchTessellation()
{
    g_triangleIndexBuffer.clear();
    g_patchCoordBuffer.clear();
    
    int maxTessLevel = g_level + g_tessLevel;
    
    unsigned int vertIndex = 0;
    g_totalPatches = 0;
    for (int i=0; i<(int)g_patchTable->GetNumPatchArrays(); ++i) {
        Far::PatchDescriptor desc = g_patchTable->GetPatchArrayDescriptor(i);
        int nControlVertices = desc.GetNumControlVertices();
            
        g_totalPatches += g_patchTable->GetNumPatches(i);
        for(int j=0; j<g_patchTable->GetNumPatches(i); j++) {
            Far::PatchTable::PatchHandle handle;

            handle.arrayIndex = i;
            handle.patchIndex = j;
            handle.vertIndex  = j * nControlVertices;
                
            Far::PatchParam patchParam = g_patchTable->GetPatchParam(i, j);
            int depth = patchParam.GetDepth();
            int tessLevel = 0;
            if( patchParam.NonQuadRoot() ) {
                tessLevel = maxTessLevel - 1 - depth;
            }
            else
                tessLevel = maxTessLevel - depth;
             
            int nSample = (1 << tessLevel) + 1;
            for(int m=0; m<nSample-1; m++) 
                for(int n=0; n<nSample-1; n++) {
                    g_triangleIndexBuffer.push_back(vertIndex + m     * nSample + n    );
                    g_triangleIndexBuffer.push_back(vertIndex + (m+1) * nSample + n    );                    
                    g_triangleIndexBuffer.push_back(vertIndex + (m+1) * nSample + n + 1);
                      
                    g_triangleIndexBuffer.push_back(vertIndex + m     * nSample + n    );
                    g_triangleIndexBuffer.push_back(vertIndex + (m+1) * nSample + n + 1);                        
                    g_triangleIndexBuffer.push_back(vertIndex + m     * nSample + n + 1);
                }
            vertIndex += nSample * nSample;  
            /*
            for(int m=0; m<nSample; m++) 
                for(int n=0; n<nSample; n++) {
                    float u = (float)m / (float)(nSample -1);
                    float v = (float)n / (float)(nSample -1);                            
                    // convert to control face level UV since PatchCoord is in that space
                    patchParam.Denormalize(u, v);
                    Osd::PatchCoord coord;
                    coord.handle = handle;
                    coord.s = u;
                    coord.t = v;
                      
                    g_patchCoordBuffer.push_back(coord);
                }   
              */  
            Osd::PatchCoord coord;
            coord.handle = handle;            
            coord.st.resize(nSample * nSample * 2);
            int id = 0;
            for(int m=0; m<nSample; m++) 
                for(int n=0; n<nSample; n++) {
                    float u = (float)m / (float)(nSample -1);
                    float v = (float)n / (float)(nSample -1);                            
                    // convert to control face level UV since PatchCoord is in that space
                    patchParam.Denormalize(u, v);

                    coord.st[id++] = u;
                    coord.st[id++] = v;
                }                
            
            g_patchCoordBuffer.push_back(coord);                             
        }
    }  
    
    g_normalBuffer.resize(vertIndex * 3);
    
    return vertIndex;
}

//------------------------------------------------------------------------------
static void
createOsdMesh(ShapeDesc const & shapeDesc, int level) {

    Shape * shape = Shape::parseObj(shapeDesc.data.c_str(), shapeDesc.scheme);

    // create Far mesh (topology)
    OpenSubdiv::Sdc::SchemeType sdctype = GetSdcType(*shape);
    OpenSubdiv::Sdc::Options sdcoptions = GetSdcOptions(*shape);

    Far::TopologyRefiner *topologyRefiner =
        OpenSubdiv::Far::TopologyRefinerFactory<Shape>::Create(*shape,
            OpenSubdiv::Far::TopologyRefinerFactory<Shape>::Options(sdctype, sdcoptions));

    g_orgPositions=shape->verts;

    delete shape;

    // save coarse topology (used for coarse mesh drawing)
    g_controlMeshDisplay.SetTopology(topologyRefiner->GetLevel(0));

    Far::StencilTable const * vertexStencils = NULL;
    int nverts=0;
    {
        // Apply feature adaptive refinement to the mesh so that we can use the
        // limit evaluation API features.
        Far::TopologyRefiner::AdaptiveOptions options(level);
        topologyRefiner->RefineAdaptive(options);

        // Generate stencil table to update the bi-cubic patches control
        // vertices after they have been re-posed (both for vertex & varying
        // interpolation)
        Far::StencilTableFactory::Options soptions;
        soptions.generateOffsets=true;
        soptions.generateIntermediateLevels=true;

        vertexStencils =
            Far::StencilTableFactory::Create(*topologyRefiner, soptions);

        // Generate bi-cubic patch table for the limit surface
        Far::PatchTableFactory::Options poptions;
        if (g_endCap == kEndCapBSplineBasis) {
            poptions.SetEndCapType(
                Far::PatchTableFactory::Options::ENDCAP_BSPLINE_BASIS);
        } else {
            poptions.SetEndCapType(
                Far::PatchTableFactory::Options::ENDCAP_GREGORY_BASIS);
        }

        Far::PatchTable const * patchTable =
            Far::PatchTableFactory::Create(*topologyRefiner, poptions);

        // append local points stencils
        if (Far::StencilTable const *localPointStencilTable =
            patchTable->GetLocalPointStencilTable()) {
            Far::StencilTable const *table =
                Far::StencilTableFactory::AppendLocalPointStencilTable(
                    *topologyRefiner, vertexStencils, localPointStencilTable);
            delete vertexStencils;
            vertexStencils = table;
        }

        // total number of vertices = coarse verts + refined verts + gregory basis verts
        nverts = vertexStencils->GetNumControlVertices() +
            vertexStencils->GetNumStencils();

        if (g_patchTable) delete g_patchTable;
        g_patchTable = patchTable;
    }

    // note that for patch eval we need coarse+refined combined buffer.
    int nCoarseVertices = topologyRefiner->GetLevel(0).GetNumVertices();

    int nSamplePoints = setupPatchTessellation();
    g_nSamplePoints = nSamplePoints;
    
    delete g_evalOutput;

    g_evalOutput = new EvalOutput<Osd::CpuVertexBuffer,
                                  Osd::CpuVertexBuffer,
                                  Far::StencilTable,
                                  Osd::CpuPatchTable,
                                  Osd::IspcEvaluator>
        (vertexStencils, NULL,
         nCoarseVertices, nverts, nSamplePoints, g_patchTable);

    const float *pVertex = g_evalOutput->GetVertexData();
    initScene(pVertex, &g_triangleIndexBuffer[0],g_triangleIndexBuffer.size() / 3, g_nSamplePoints);
    
    updateGeom();

    // compute model bounding
    float min[3] = { FLT_MAX,  FLT_MAX,  FLT_MAX};
    float max[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (size_t i=0; i <g_orgPositions.size()/3; ++i) {
        for(int j=0; j<3; ++j) {
            float v = g_orgPositions[i*3+j];
            min[j] = std::min(min[j], v);
            max[j] = std::max(max[j], v);
        }
    }
    for (int j=0; j<3; ++j) {
        g_center[j] = (min[j] + max[j]) * 0.5f;
        g_size += (max[j]-min[j])*(max[j]-min[j]);
    }
    g_size = sqrtf(g_size);
    
    delete topologyRefiner;
}

//------------------------------------------------------------------------------
static void
fitFrame() {

    g_pan[0] = g_pan[1] = 0;
    g_dolly = g_size;
}

inline void
applyRotation(float *v, const float *m)
{
    float r[3];
    r[0] = v[0] * m[0] + v[1] * m[4] + v[2] * m[8];
    r[1] = v[0] * m[1] + v[1] * m[5] + v[2] * m[9];
    r[2] = v[0] * m[2] + v[1] * m[6] + v[2] * m[10];
    v[0] = r[0];
    v[1] = r[1];
    v[2] = r[2];
}

void renderScene(unsigned int *buffer, int width, int length,
                 float *eye, float *xaxis, float *yaxis, float *zaxis,
                 float fov,  float znear, bool shadowOn);
                 
//------------------------------------------------------------------------------
static void
display() {
    Stopwatch s;
    s.Start();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glViewport(0, 0, g_width, g_height);
    g_hud.FillBackground();
    
    // prepare view matrix
    double aspect = g_width/(double)g_height;
    identity(g_transformData.ModelViewMatrix);
    translate(g_transformData.ModelViewMatrix, -g_pan[0], -g_pan[1], -g_dolly);
    rotate(g_transformData.ModelViewMatrix, g_rotate[1], 1, 0, 0);
    rotate(g_transformData.ModelViewMatrix, g_rotate[0], 0, 1, 0);
    rotate(g_transformData.ModelViewMatrix, -90, 1, 0, 0);
    translate(g_transformData.ModelViewMatrix,
              -g_center[0], -g_center[1], -g_center[2]);
    inverseMatrix(g_transformData.ModelViewInverseMatrix,
                  g_transformData.ModelViewMatrix);              
    perspective(g_transformData.ProjectionMatrix,
                45.0f, (float)aspect, 0.1f, 500.0f);
             
             
    memset(g_frameBuffer, 255, g_width * g_height * sizeof(unsigned int));    
    
    float eye[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    apply(eye, g_transformData.ModelViewInverseMatrix);
    float vx[3] = {1.0f, 0.0f, 0.0f};
    applyRotation(vx, g_transformData.ModelViewInverseMatrix);
    float vy[3] = {0.0f, 1.0f, 0.0f};
    applyRotation(vy, g_transformData.ModelViewInverseMatrix);
    float vz[3] = {0.0f, 0.0f, 1.0f};
    applyRotation(vz, g_transformData.ModelViewInverseMatrix);      
    renderScene(g_frameBuffer, g_width, g_height, eye, vx, vy, vz, 45.0f, 0.1f, g_shadow);
    
    glDisable(GL_DEPTH_TEST);
    
    glMatrixMode(GL_MODELVIEW); 
    glLoadIdentity();    
    glMatrixMode(GL_PROJECTION); 
    glLoadIdentity();     
    glRasterPos2i(-1, -1);
    glDrawPixels(g_width, g_height, GL_RGBA, GL_UNSIGNED_BYTE, g_frameBuffer); 
    
    glEnable(GL_DEPTH_TEST);
    /*             
    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixf(g_transformData.ModelViewMatrix);
    
    glMatrixMode(GL_PROJECTION);
    glLoadMatrixf(g_transformData.ProjectionMatrix);     
        
    const float *pVertex = g_evalOutput->GetVertexData();
    glColor3f(1.0, 0.0, 0.0);
    glPointSize(4);
    glBegin(GL_POINTS);
    for(int i=0; i<g_nSamplePoints; i++) {
        glVertex3f(pVertex[i*3],
                   pVertex[i*3+1],
                   pVertex[i*3+2]);
    }
    glEnd();
    */

    g_fpsTimer.Stop();
    float elapsed = (float)g_fpsTimer.GetElapsed();
    if (not g_freeze) {
        g_animTime += elapsed;
    }
    g_fpsTimer.Start();

    if (g_hud.IsVisible()) {

        typedef OpenSubdiv::Far::PatchDescriptor Descriptor;

        double fps = 1.0/elapsed;

        int y = -180;
        g_hud.DrawString(10, y, "Tess level  : %d", g_tessLevel); y+= 20;
        g_hud.DrawString(10, y, "Patches     : %d", g_totalPatches); y+= 20;
        g_hud.DrawString(10, y, "Triangles   : %d", g_triangleIndexBuffer.size()/3); y+= 20;
        g_hud.DrawString(10, y, "Vertices    : %d", g_nSamplePoints); y+= 20;
        g_hud.DrawString(10, y, "Render time : %.3f ms", elapsed); y+= 20;
        g_hud.DrawString(10, y, "FPS         : %3.1f", fps); y+= 20;

        g_hud.Flush();
    }


    glFinish();

    GLUtils::CheckGLErrors("display leave\n");
}

//------------------------------------------------------------------------------
static void
motion(GLFWwindow *, double dx, double dy) {

    int x=(int)dx, y=(int)dy;

    if (g_hud.MouseCapture()) {
        // check gui
        g_hud.MouseMotion(x, y);
    } else if (g_mbutton[0] && !g_mbutton[1] && !g_mbutton[2]) {
        // orbit
        g_rotate[0] += x - g_prev_x;
        g_rotate[1] += y - g_prev_y;
    } else if (!g_mbutton[0] && !g_mbutton[1] && g_mbutton[2]) {
        // pan
        g_pan[0] -= g_dolly*(x - g_prev_x)/g_width;
        g_pan[1] += g_dolly*(y - g_prev_y)/g_height;
    } else if ((g_mbutton[0] && !g_mbutton[1] && g_mbutton[2]) or
               (!g_mbutton[0] && g_mbutton[1] && !g_mbutton[2])) {
        // dolly
        g_dolly -= g_dolly*0.01f*(x - g_prev_x);
        if(g_dolly <= 0.01) g_dolly = 0.01f;
    }

    g_prev_x = x;
    g_prev_y = y;
}

//------------------------------------------------------------------------------
static void
mouse(GLFWwindow *, int button, int state, int /* mods */) {

    if (state == GLFW_RELEASE)
        g_hud.MouseRelease();

    if (button == 0 && state == GLFW_PRESS && g_hud.MouseClick(g_prev_x, g_prev_y))
        return;

    if (button < 3) {
        g_mbutton[button] = (state == GLFW_PRESS);
    }
}

//------------------------------------------------------------------------------
static void
uninitGL() {
}

//------------------------------------------------------------------------------
static void
reshape(GLFWwindow *, int width, int height) {

    g_width = width;
    g_height = height;

    int windowWidth = g_width, windowHeight = g_height;

    // window size might not match framebuffer size on a high DPI display
    glfwGetWindowSize(g_window, &windowWidth, &windowHeight);

    g_hud.Rebuild(windowWidth, windowHeight, width, height);
}

//------------------------------------------------------------------------------
void windowClose(GLFWwindow*) {
    g_running = false;
}

//------------------------------------------------------------------------------
static void
toggleFullScreen() {
    // XXXX manuelk : to re-implement from glut
}

//------------------------------------------------------------------------------
static void
keyboard(GLFWwindow *, int key, int /* scancode */, int event, int /* mods */) {

    if (event == GLFW_RELEASE) return;
    if (g_hud.KeyDown(tolower(key))) return;

    switch (key) {
        case 'Q': g_running = 0; break;
        case 'F': fitFrame(); break;
        case GLFW_KEY_TAB: toggleFullScreen(); break;
        case '+':
        case '=':  g_tessLevel++; 
                   createOsdMesh(g_defaultShapes[g_currentShape], g_level);
                   break;
        case '-':  g_tessLevel = std::max(g_tessLevelMin, g_tessLevel-1); 
                   createOsdMesh(g_defaultShapes[g_currentShape], g_level);
                   break;
        case GLFW_KEY_ESCAPE: g_hud.SetVisible(!g_hud.IsVisible()); break;
        case 'X': GLUtils::WriteScreenshot(g_width, g_height); break;
    }
}

static void
callbackEndCap(int endCap) {
    g_endCap = endCap;
    createOsdMesh(g_defaultShapes[g_currentShape], g_level);
}

static void
callbackLevel(int l) {
    g_level = l;
    createOsdMesh(g_defaultShapes[g_currentShape], g_level);
}

static void
callbackModel(int m) {
    if (m < 0)
        m = 0;

    if (m >= (int)g_defaultShapes.size())
        m = (int)g_defaultShapes.size() - 1;

    g_currentShape = m;
    createOsdMesh(g_defaultShapes[g_currentShape], g_level);
}

static void
callbackCheckBox(bool checked, int button) {

    switch (button) {
    case kHUD_CB_ANIMATE_VERTICES:
        g_moveScale = checked;
        break;
    case kHUD_CB_FREEZE:
        g_freeze = checked;        
        break;
    case kHUD_CB_SHADOW:
        g_shadow = checked;
        break;        
    }
}

static void
initHUD() {
    int windowWidth = g_width, windowHeight = g_height;
    int frameBufferWidth = g_width, frameBufferHeight = g_height;

    // window size might not match framebuffer size on a high DPI display
    glfwGetWindowSize(g_window, &windowWidth, &windowHeight);
    glfwGetFramebufferSize(g_window, &frameBufferWidth, &frameBufferHeight);

    g_hud.Init(windowWidth, windowHeight, frameBufferWidth, frameBufferHeight);

    int y = 10;
    /*
    g_hud.AddCheckBox("Control edges (H)",
                      g_controlMeshDisplay.GetEdgesDisplay(),
                      10, y, callbackCheckBox,
                      kHUD_CB_DISPLAY_CONTROL_MESH_EDGES, 'h');
    y += 20;
    g_hud.AddCheckBox("Control vertices (J)",
                      g_controlMeshDisplay.GetVerticesDisplay(),
                      10, y, callbackCheckBox,
                      kHUD_CB_DISPLAY_CONTROL_MESH_VERTS, 'j');

    y += 20;    
    */
    g_hud.AddCheckBox("Animate vertices (M)", g_moveScale != 0,
                      10, y, callbackCheckBox, kHUD_CB_ANIMATE_VERTICES, 'm');
    y += 20;
    g_hud.AddCheckBox("Freeze (spc)", g_freeze != 0,
                      10, y, callbackCheckBox, kHUD_CB_FREEZE, ' ');
    y += 20;

    g_hud.AddCheckBox("Shadow on", g_shadow != 0,
                      10, y, callbackCheckBox, kHUD_CB_SHADOW, ' ');
    y += 20;

    for (int i = 1; i < 11; ++i) {
        char level[16];
        sprintf(level, "Lv. %d", i);
        g_hud.AddRadioButton(3, level, i==g_level, 10, 80+i*20, callbackLevel, i, '0'+(i%10));
    }

    int shapes_pulldown = g_hud.AddPullDown("Shape (N)", -300, 10, 300, callbackModel, 'n');
    for (int i = 0; i < (int)g_defaultShapes.size(); ++i) {
        g_hud.AddPullDownButton(shapes_pulldown, g_defaultShapes[i].name.c_str(),i);
    }

    g_hud.Rebuild(windowWidth, windowHeight, frameBufferWidth, frameBufferHeight);
}

//------------------------------------------------------------------------------
static void
initGL() {
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glCullFace(GL_BACK);
    glEnable(GL_CULL_FACE);
}

//------------------------------------------------------------------------------
static void
idle() {

    if (not g_freeze) {
        g_frame++;
        updateGeom();
    }

    if (g_repeatCount != 0 and g_frame >= g_repeatCount)
        g_running = 0;
}

//------------------------------------------------------------------------------
static void
callbackErrorOsd(OpenSubdiv::Far::ErrorType err, const char *message) {
    printf("Error: %d\n", err);
    printf("%s", message);
}

//------------------------------------------------------------------------------
static void
callbackErrorGLFW(int error, const char* description) {
    fprintf(stderr, "GLFW Error (%d) : %s\n", error, description);
}
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
int main(int argc, char ** argv) {

    bool fullscreen = false;
    std::string str;
    std::vector<char const *> animobjs;

    for (int i = 1; i < argc; ++i) {
        if (strstr(argv[i], ".obj")) {
            animobjs.push_back(argv[i]);
        }
        else if (!strcmp(argv[i], "-axis")) {
            g_axis = false;
        }
        else if (!strcmp(argv[i], "-d")) {
            g_level = atoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "-c")) {
            g_repeatCount = atoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "-f")) {
            fullscreen = true;
        }
        else {
            std::ifstream ifs(argv[1]);
            if (ifs) {
                std::stringstream ss;
                ss << ifs.rdbuf();
                ifs.close();
                str = ss.str();
                g_defaultShapes.push_back(ShapeDesc(argv[1], str.c_str(), kCatmark));
            }
        }
    }

    if (not animobjs.empty()) {

        g_defaultShapes.push_back(ShapeDesc(animobjs[0], "", kCatmark));

        g_objAnim = ObjAnim::Create(animobjs, g_axis);
    }

    initShapes();

    g_fpsTimer.Start();

    OpenSubdiv::Far::SetErrorCallback(callbackErrorOsd);

    glfwSetErrorCallback(callbackErrorGLFW);
    if (not glfwInit()) {
        printf("Failed to initialize GLFW\n");
        return 1;
    }

    static const char windowTitle[] = "OpenSubdiv embreeViewer " OPENSUBDIV_VERSION_STRING;

    GLUtils::SetMinimumGLVersion(argc, argv);

    if (fullscreen) {

        g_primary = glfwGetPrimaryMonitor();

        // apparently glfwGetPrimaryMonitor fails under linux : if no primary,
        // settle for the first one in the list
        if (not g_primary) {
            int count = 0;
            GLFWmonitor ** monitors = glfwGetMonitors(&count);

            if (count)
                g_primary = monitors[0];
        }

        if (g_primary) {
            GLFWvidmode const * vidmode = glfwGetVideoMode(g_primary);
            g_width = vidmode->width;
            g_height = vidmode->height;
        }
    }

    g_window = glfwCreateWindow(g_width, g_height, windowTitle,
        fullscreen and g_primary ? g_primary : NULL, NULL);

    if (not g_window) {
        std::cerr << "Failed to create OpenGL context.\n";
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(g_window);
    GLUtils::PrintGLVersion();

    // accommocate high DPI displays (e.g. mac retina displays)
    glfwGetFramebufferSize(g_window, &g_width, &g_height);
    glfwSetFramebufferSizeCallback(g_window, reshape);

    glfwSetKeyCallback(g_window, keyboard);
    glfwSetCursorPosCallback(g_window, motion);
    glfwSetMouseButtonCallback(g_window, mouse);
    glfwSetWindowCloseCallback(g_window, windowClose);

#if defined(OSD_USES_GLEW)
#ifdef CORE_PROFILE
    // this is the only way to initialize glew correctly under core profile context.
    glewExperimental = true;
#endif
    if (GLenum r = glewInit() != GLEW_OK) {
        printf("Failed to initialize glew. Error = %s\n", glewGetErrorString(r));
        exit(1);
    }
#ifdef CORE_PROFILE
    // clear GL errors which was generated during glewInit()
    glGetError();
#endif
#endif

    initGL();

    glfwSwapInterval(0);

    initHUD();
    createOsdMesh(g_defaultShapes[g_currentShape], g_level);

    // create frame buffer for embree rendering
    g_frameBuffer = (unsigned int *)malloc(g_width * g_height * sizeof(unsigned int));
    
    while (g_running) {
        idle();
        display();

        glfwPollEvents();
        glfwSwapBuffers(g_window);

        glFinish();
    }

    uninitGL();
    glfwTerminate();
}

//------------------------------------------------------------------------------
