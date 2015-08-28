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

#include <cfloat>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

int g_width  = 800;
int g_height = 800;
unsigned int *g_frameBuffer = NULL;
int g_running = true;
                 
//------------------------------------------------------------------------------
static void
display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glViewport(0, 0, g_width, g_height);
        
    glDisable(GL_DEPTH_TEST);
    
    glMatrixMode(GL_MODELVIEW); 
    glLoadIdentity();    
    glMatrixMode(GL_PROJECTION); 
    glLoadIdentity();     
    glRasterPos2i(-1, -1);
    glDrawPixels(g_width, g_height, GL_RGBA, GL_UNSIGNED_BYTE, g_frameBuffer); 
    
    glEnable(GL_DEPTH_TEST);

    glFinish();
}

//------------------------------------------------------------------------------
void windowClose(GLFWwindow*) {
    g_running = false;
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


    glfwSetErrorCallback(callbackErrorGLFW);
    if (not glfwInit()) {
        printf("Failed to initialize GLFW\n");
        return 1;
    }

    static const char windowTitle[] = "OpenSubdiv embreeViewer " OPENSUBDIV_VERSION_STRING;

    GLUtils::SetMinimumGLVersion(argc, argv);

    g_window = glfwCreateWindow(g_width, g_height, windowTitle,
        fullscreen and g_primary ? g_primary : NULL, NULL);

    if (not g_window) {
        std::cerr << "Failed to create OpenGL context.\n";
        glfwTerminate();
        return 1;
    }

    glfwSetWindowCloseCallback(g_window, windowClose);
    glfwMakeContextCurrent(g_window);
    GLUtils::PrintGLVersion();

    // accommocate high DPI displays (e.g. mac retina displays)
    glfwGetFramebufferSize(g_window, &g_width, &g_height);

    initGL();

    glfwSwapInterval(0);

    // create frame buffer for embree rendering
    g_frameBuffer = (unsigned int *)malloc(g_width * g_height * sizeof(unsigned int));
    FILE *fp = fopen(argv[1], "r");
    fread(g_frameBuffer, g_width * g_height * sizeof(unsigned int), 1, fp);
    fclose(fp);
    
    while (g_running) {
        idle();
        display();

        glfwPollEvents();
        glfwSwapBuffers(g_window);

        glFinish();
    }

    glfwTerminate();
}

//------------------------------------------------------------------------------
