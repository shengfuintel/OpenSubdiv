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

#include <far/error.h>

#include <osd/cpuEvaluator.h>
#include <osd/ispcEvaluator.h>
#include <osd/cpuVertexBuffer.h>
#include <osd/cpuPatchTable.h>
#include <osd/mesh.h>
#include <osd/ispcEvalLimitKernel.isph>

using namespace OpenSubdiv;

#include </opt/intel/vtune_amplifier_xe/include/ittnotify.h>

#include "../../regression/common/far_utils.h"
#include "../common/simple_math.h"

#include <tbb/parallel_for.h>
#include <tbb/tick_count.h>
#include <tbb/task_scheduler_init.h>

#include <cfloat>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

//#define RNEDER_SCENE

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

enum EndCap      { kEndCapNone = 0,
                   kEndCapBSplineBasis,
                   kEndCapGregoryBasis,
                   kEndCapLegacyGregory };

int g_currentShape = 0;

int   g_width = 800,
      g_height = 800;
float g_center[3] = {0, 0, 0};
float g_size = 0;
float g_rotate[2] = {-87, 31},
      g_dolly = 5,
      g_pan[2] = {0, 0};
      
unsigned int *g_frameBuffer = NULL;

// geometry
std::vector<float> g_orgPositions;

Far::PatchTable const *g_patchTable = NULL;

// tessellated mesh
std::vector<unsigned int>    g_triangleIndexBuffer;
std::vector<float>           g_normalBuffer;
std::vector<Osd::PatchCoord> g_patchCoordBuffer;
int g_nSamplePoints = 0;
int g_totalPatches = 0;

int g_level = 4;
int g_tessLevel = 4;
int g_tessLevelMin = 1;
float g_moveScale = 0.0f;

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
    /*
        EVALUATOR const *evalInstance = OpenSubdiv::Osd::GetEvaluator<EVALUATOR>(
            _evaluatorCache, _srcDesc, _vertexDesc, _duDesc, _dvDesc, _deviceContext);
        EVALUATOR::EvalPatches(
            _srcData, _srcDesc,
            _vertexData, _vertexDesc,
            _derivatives, _duDesc,
            _derivatives, _dvDesc,
            g_patchCoordBuffer.size(),
            &g_patchCoordBuffer[0],
            _patchTable, evalInstance, _deviceContext);
            */
        // Copy BufferDescriptor to ispc version
        // Since memory alignment in ISPC may be different from C++,
        // we use the assignment for each field instead of the assignment for 
        // the whole struct
        //__itt_resume();
        
        ispc::BufferDescriptor ispcSrcDesc;
        ispcSrcDesc.length = _srcDesc.length;
        ispcSrcDesc.stride = _srcDesc.stride;      
        ispcSrcDesc.offset = _srcDesc.offset;                                     

                    
        float *src = _srcData->BindCpuBuffer();
        float *dev = _derivatives->BindCpuBuffer();
        float *dst = _vertexData->BindCpuBuffer();
        
        const Osd::PatchParam *_patchParamBuffer = _patchTable->GetPatchParamBuffer();
        const int *_patchIndexBuffer  = _patchTable->GetPatchIndexBuffer();
        const Osd::PatchArray *_patchArray = _patchTable->GetPatchArrayBuffer();
        printf("coord size = %d\n", (int)g_patchCoordBuffer.size());
        tbb::blocked_range<int> range = tbb::blocked_range<int>(0, g_patchCoordBuffer.size());
        tbb::parallel_for(range, [&](const tbb::blocked_range<int> &r)
        {            
//          for(int i=0; i<g_patchCoordBuffer.size(); i++) {
            ispc::BufferDescriptor ispcDstDesc, ispcDuDesc, ispcDvDesc;
            ispcDstDesc.length = _vertexDesc.length;
            ispcDstDesc.stride = _vertexDesc.stride;
    
            ispcDuDesc.length  = _duDesc.length;
            ispcDuDesc.stride  = _duDesc.stride;
    
            ispcDvDesc.length  = _dvDesc.length;
            ispcDvDesc.stride  = _dvDesc.stride;
            
            for (uint i=r.begin(); i < r.end(); i++) {
                ispcDstDesc.offset = _vertexDesc.offset + g_patchCoordBuffer[i].offset * _vertexDesc.stride;
                ispcDuDesc.offset  = _duDesc.offset  + g_patchCoordBuffer[i].offset * _duDesc.stride;               
                ispcDvDesc.offset  = _dvDesc.offset  + g_patchCoordBuffer[i].offset * _dvDesc.stride;          
              
                const Far::PatchTable::PatchHandle &handle = g_patchCoordBuffer[i].handle;              
                Osd::PatchArray const &array = _patchArray[handle.arrayIndex];
                int patchType = array.GetPatchType();
                Far::PatchParam const & param = _patchParamBuffer[handle.patchIndex];

                unsigned int bitField = param.field1;

                const int *cvs = &_patchIndexBuffer[array.indexBase + handle.vertIndex];

                int nCoord = g_patchCoordBuffer[i].st.size() / 2;
                __declspec( align(64) ) float u[nCoord];
                __declspec( align(64) ) float v[nCoord];        
        
                for(int n=0; n<nCoord; n++) {
                    u[n] = g_patchCoordBuffer[i].st[2*n  ];
                    v[n] = g_patchCoordBuffer[i].st[2*n+1];            
                }
        
                if (patchType == Far::PatchDescriptor::REGULAR) {
                    ispc::evalBSpline(bitField, nCoord, u, v, cvs, ispcSrcDesc, src, 
                              ispcDstDesc, dst, ispcDuDesc, dev, ispcDvDesc, dev);
                } else if (patchType == Far::PatchDescriptor::GREGORY_BASIS) {
                     ispc::evalGregory(bitField, nCoord, u, v, cvs, ispcSrcDesc, src, 
                               ispcDstDesc, dst, ispcDuDesc, dev, ispcDvDesc, dev);        
                } else if (patchType == Far::PatchDescriptor::QUADS) {
                     ispc::evalBilinear(bitField, nCoord, u, v, cvs, ispcSrcDesc, src, 
                               ispcDstDesc, dst, ispcDuDesc, dev, ispcDvDesc, dev);           
                } else {
                    assert(0);
                }             
            }
        });          
        
        //__itt_pause();
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
        /*
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
        */
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

EvalOutputBase        *g_evalOutput = NULL;

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
    for (int i = 0; i < nverts; ++i) {
        vertex.push_back( p[0]);
        vertex.push_back( p[1]);
        vertex.push_back( p[2]);
        p += 3;
    }

    // Run Compute pass to pose the control vertices ---------------------------

    // update coarse vertices
    g_evalOutput->UpdateData(&vertex[0], 0, nverts);

    // Refine
    tbb::tick_count start = tbb::tick_count::now();
    g_evalOutput->Refine();
    tbb::tick_count end = tbb::tick_count::now();
    printf("subdivision time = %g\n", (end - start).seconds());

    // update patchcoord to be evaluated
    g_evalOutput->UpdatePatchCoords(g_patchCoordBuffer);

    // evaluate positions
    start = tbb::tick_count::now();
    g_evalOutput->EvalPatchesWithDerivatives();
    end = tbb::tick_count::now();
    printf("patch evalation time = %g\n", (end - start).seconds());
    
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
             
            int nSample = (1 << tessLevel);
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
            //printf("nSample = %d\n", nSample);
            
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
    
    printf("total number of patches = %d\n", g_totalPatches);
    
    int offset = 0;
    for(int i=0; i<g_patchCoordBuffer.size(); i++) {
        g_patchCoordBuffer[i].offset = offset;
        offset += g_patchCoordBuffer[i].st.size() / 2;
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
//        poptions.SetEndCapType(
//                Far::PatchTableFactory::Options::ENDCAP_GREGORY_BASIS);
        poptions.SetEndCapType(
                Far::PatchTableFactory::Options::ENDCAP_BSPLINE_BASIS);
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
    
    updateGeom();
        
#ifdef RNEDER_SCENE        
    const float *pVertex = g_evalOutput->GetVertexData();
    initScene(pVertex, &g_triangleIndexBuffer[0],g_triangleIndexBuffer.size() / 3, g_nSamplePoints);
    
    FILE *fp = fopen("vertex_buffer", "w");
    fwrite(pVertex, g_nSamplePoints * sizeof(float) * 3, 1, fp);
    fclose(fp);

    fp = fopen("index_buffer", "w");
    fwrite(&g_triangleIndexBuffer[0], g_triangleIndexBuffer.size() * sizeof(int), 1, fp);
    fclose(fp);
    updateScene(pVertex, &g_triangleIndexBuffer[0], &g_normalBuffer[0], 
                g_triangleIndexBuffer.size()/3, g_nSamplePoints);
#endif                
                
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
    
    //delete topologyRefiner;
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
    
             
    memset(g_frameBuffer, 255, g_width * g_height * sizeof(unsigned int));    
    identity(g_transformData.ModelViewMatrix);
    translate(g_transformData.ModelViewMatrix, -g_pan[0], -g_pan[1], -g_dolly);
    rotate(g_transformData.ModelViewMatrix, g_rotate[1], 1, 0, 0);
    rotate(g_transformData.ModelViewMatrix, g_rotate[0], 0, 1, 0);
    rotate(g_transformData.ModelViewMatrix, -90, 1, 0, 0);
    translate(g_transformData.ModelViewMatrix,
              -g_center[0], -g_center[1], -g_center[2]);
    inverseMatrix(g_transformData.ModelViewInverseMatrix,
                  g_transformData.ModelViewMatrix);              
                      
    float eye[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    apply(eye, g_transformData.ModelViewInverseMatrix);
    float vx[3] = {1.0f, 0.0f, 0.0f};
    applyRotation(vx, g_transformData.ModelViewInverseMatrix);
    float vy[3] = {0.0f, 1.0f, 0.0f};
    applyRotation(vy, g_transformData.ModelViewInverseMatrix);
    float vz[3] = {0.0f, 0.0f, 1.0f};
    applyRotation(vz, g_transformData.ModelViewInverseMatrix);      
    
#ifdef RNEDER_SCENE    
    renderScene(g_frameBuffer, g_width, g_height, eye, vx, vy, vz, 45.0f, 0.1f, 0);
#endif    
}

//------------------------------------------------------------------------------
static void
callbackErrorOsd(OpenSubdiv::Far::ErrorType err, const char *message) {
    printf("Error: %d\n", err);
    printf("%s", message);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
int main(int argc, char ** argv) {
    if(argc < 2) {
        printf("Usage: performance nThread\n");
        exit(-1);
    }
        
    int nThread = atoi(argv[1]);

    tbb::task_scheduler_init init(nThread);

    initShapes();

    OpenSubdiv::Far::SetErrorCallback(callbackErrorOsd);

    createOsdMesh(g_defaultShapes[g_currentShape], g_level);
    
    // create frame buffer for embree rendering
    g_frameBuffer = (unsigned int *)malloc(g_width * g_height * sizeof(unsigned int));
    
    display();
    
    FILE *fp = fopen("image.bin", "w");
    fwrite(g_frameBuffer, g_width * g_height * sizeof(unsigned int), 1, fp);
    fclose(fp);
}

//------------------------------------------------------------------------------
