#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2011 Nicolas P. Rougier (Nicolas.Rougier@inria.fr)
#
# Gray-Scott reation diffusion simulation
#
# This software is governed by the CeCILL license under French law and abiding
# by the rules of distribution of free software. You can use, modify and/ or
# redistribute the software under the terms of the CeCILL license as circulated
# by CEA, CNRS and INRIA at the following URL http://www.cecill.info
#
# As a counterpart to the access to the source code and rights to copy, modify
# and redistribute granted by the license, users are provided only with a
# limited warranty and the software's author, the holder of the economic
# rights, and the successive licensors have only limited liability.
#
# In this respect, the user's attention is drawn to the risks associated with
# loading, using, modifying and/or developing or reproducing the software by
# the user in light of its specific status of free software, that may mean that
# it is complicated to manipulate, and that also therefore means that it is
# reserved for developers and experienced professionals having in-depth
# computer knowledge. Users are therefore encouraged to load and test the
# software's suitability as regards their requirements in conditions enabling
# the security of their systems and/or data to be ensured and, more generally,
# to use and operate it in the same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# -----------------------------------------------------------------------------
'''
Reaction Diffusion model:

"Numerical simulations of a simple reaction-diffusion model reveal a surprising
 variety of irregular spatiotemporal patterns. These patterns arise in response
 to finite-amplitude perturbations. Some  of them resemble the steady irregular
 patterns recently observed in thin  gel reactor experiments. Others consist of
 spots that grow until they reach a critical size, at which time they divide in
 two. If in some region the spots  become overcrowded, all of the spots in that
 region decay into the uniform background."
                                                                John E. Pearson
                                            Complex Patterns in a Simple System
                                               Science 261, 5118, 189-192, 1993

Equations
---------
∂u/∂t = rᵤ∇²u - uv² + f(1-u)
∂v/∂t = rᵥ∇²v + uv² - (f+k)v

Chemical reactions
------------------
U + 2V → 3V
     V → P

Variables & parameters
----------------------
U and V and P are chemical species. 
u and v represent their concentrations. 
rᵤ and rᵥ are their diffusion rates. 
k represents the rate of conversion of V to P. 
f represents the rate of the process that feeds U and drains U,V and P

References
----------
"Complex Patterns in a Simple System"
 J.E. Pearson
 Science 261, 5118, 189-192, 1993.

"Pattern Formation by Interacting Chemical Fronts"
 K.J. Lee, W.D. McCormick, Qi Ouyang, and H.L. Swinney
 Science 261, 5118, 192-194, 1993.
'''
import ctypes
import pyglet
pyglet.options['debug_gl'] = False
import numpy as np
import pyglet.gl as gl
from shader import Shader


if __name__ == '__main__':

    # Screen information
    # ------------------
    platform = pyglet.window.get_platform()
    display  = platform.get_default_display()
    screens  = display.get_screens()
    screen   = screens[0]

    # Parameters
    # ----------
    scale = 4
    width, height = screen.width//scale, screen.height//scale
    dt = 1.0
    dd = 1.5
    species = {
        # name : [r_u, r_v, f, k]
        'Bacteria 1'    : [0.16, 0.08, 0.035, 0.065],
        'Bacteria 2'    : [0.14, 0.06, 0.035, 0.065],
        'Coral'         : [0.16, 0.08, 0.060, 0.062],
        'Fingerprint'   : [0.19, 0.05, 0.060, 0.062],
        'Spirals'       : [0.10, 0.10, 0.018, 0.050],
        'Spirals Dense' : [0.12, 0.08, 0.020, 0.050],
        'Spirals Fast'  : [0.10, 0.16, 0.020, 0.050],
        'Unstable'      : [0.16, 0.08, 0.020, 0.055],
        'Worms 1'       : [0.16, 0.08, 0.050, 0.065],
        'Worms 2'       : [0.16, 0.08, 0.054, 0.063],
        'Zebrafish'     : [0.16, 0.08, 0.035, 0.060]
    }


    # Window creation
    window = pyglet.window.Window(fullscreen=True,
                                  caption='Gray-Scott Reaction Diffusion',
                                  visible = False, vsync=False, resizable=True)
    window.set_location(window.screen.width/2  - window.width/2,
                        window.screen.height/2 - window.height/2)


    # texture_s holds species
    # -----------------------
    P = np.zeros((height,width,4), dtype=np.float32)
    P[:,:] = species['Bacteria 2']
    data = P.ctypes.data
    texture_s = pyglet.image.Texture.create_for_size(
        gl.GL_TEXTURE_2D, width, height, gl.GL_RGBA32F_ARB)
    gl.glBindTexture(texture_s.target, texture_s.id)
    gl.glTexImage2D(texture_s.target, texture_s.level, gl.GL_RGBA32F_ARB,
                    width, height, 0, gl.GL_RGBA, gl.GL_FLOAT, data)
    gl.glBindTexture(texture_s.target, 0)
    
    # texture_uv holds U & V values (red and green channels)
    # ------------------------------------------------------
    UV = np.zeros((height,width,4), dtype=np.float32)
    UV[:,:,0] = 1.0
    r = 32
    UV[height/2-r:height/2+r, width/2-r:width/2+r, 0] = 0.50
    UV[height/2-r:height/2+r, width/2-r:width/2+r, 1] = 0.25
    data = UV.ctypes.data
    texture_uv = pyglet.image.Texture.create_for_size(
        gl.GL_TEXTURE_2D, width, height, gl.GL_RGBA32F)
    gl.glBindTexture(texture_uv.target, texture_uv.id)
    gl.glTexImage2D(texture_uv.target, texture_uv.level, gl.GL_RGBA32F_ARB,
                    width, height, 0, gl.GL_RGBA, gl.GL_FLOAT, data)
    gl.glBindTexture(texture_uv.target, 0)


    # texture_p holds pointer mask
    # ----------------------------
    D = np.ones((8,8,4),dtype=np.ubyte)*0
    #D[4:12:,4:12] = 0
    image = pyglet.image.ImageData(D.shape[1],D.shape[0],'RGBA',D.ctypes.data)
    sprite = pyglet.sprite.Sprite(image)


    # Reaction-diffusion shader
    # -------------------------
    vertex_shader   = open('./reaction-diffusion.vert').read()
    fragment_shader = open('./reaction-diffusion.frag').read()
    reaction_shader = Shader(vertex_shader, fragment_shader)

    reaction_shader.bind()
    reaction_shader.uniformi('texture', 0)
    reaction_shader.uniformi('params',  1)
    reaction_shader.uniformi('display', 2)
    reaction_shader.uniformf('dt', dt)
    reaction_shader.uniformf('dx', 1.0/width)
    reaction_shader.uniformf('dy', 1.0/height)
    reaction_shader.uniformf('dd', dd)
    reaction_shader.unbind()

    # Color shader
    # ------------
    vertex_shader   = open('./color.vert').read()
    fragment_shader = open('./color.frag').read()
    color_shader    = Shader(vertex_shader, fragment_shader)

    color_shader.bind()
    color_shader.uniformi('texture', 0)
    color_shader.unbind()

    # Framebuffer
    # -----------
    framebuffer = gl.GLuint(0)
    gl.glGenFramebuffersEXT(1, ctypes.byref(framebuffer))
    gl.glBindFramebufferEXT(gl.GL_FRAMEBUFFER_EXT, framebuffer)
    gl.glFramebufferTexture2DEXT(gl.GL_FRAMEBUFFER_EXT, gl.GL_COLOR_ATTACHMENT0_EXT,
                                 gl.GL_TEXTURE_2D, texture_uv.id, 0); 
    gl.glBindFramebufferEXT(gl.GL_FRAMEBUFFER_EXT, 0)



    @window.event
    def on_mouse_drag(x, y, dx, dy, button, modifiers):
        sprite.x = int((x/float(window.width)) * width)-sprite.width//2
        sprite.y = int((y/float(window.height)) * height)-sprite.height//2

        # Compute
        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0, width, 0, height, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glBindFramebufferEXT(gl.GL_FRAMEBUFFER_EXT, framebuffer)
        sprite.draw()
        gl.glBindFramebufferEXT(gl.GL_FRAMEBUFFER_EXT, 0)

        

    @window.event
    def on_draw():
        gl.glClearColor(1.0,1.0,1.0,1.0)
	window.clear()

        # Compute
        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0, 1, 0, 1, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)

        gl.glActiveTexture( gl.GL_TEXTURE1 )
	gl.glBindTexture(texture_s.target, texture_s.id)

        gl.glActiveTexture( gl.GL_TEXTURE0 )
	gl.glBindTexture(texture_uv.target, texture_uv.id)

        gl.glBindFramebufferEXT(gl.GL_FRAMEBUFFER_EXT, framebuffer)
	reaction_shader.bind()
        texture_uv.blit(x=0.0, y=0.0, width=1.0, height=1.0)
	reaction_shader.unbind()
        gl.glBindFramebufferEXT(gl.GL_FRAMEBUFFER_EXT, 0)

        # Render
        gl.glViewport(0, 0, window.width, window.height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0, 1, 0, 1, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)

	color_shader.bind()
        texture_uv.blit(x=0.0, y=0.0, width=1.0, height=1.0)
	color_shader.bind()


#pyglet.clock.schedule_interval(lambda dt: None, 1.0/60.0)
pyglet.clock.schedule(lambda dt: None)
window.set_visible(True)
pyglet.app.run()
