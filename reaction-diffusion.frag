// -----------------------------------------------------------------------------
// Copyright 2011 Nicolas P. Rougier (Nicolas.Rougier@inria.fr)
//
// Gray-Scott reation diffusion simulation
//
// This software is governed by the CeCILL license under French law and abiding
// by the rules of distribution of free software. You can use, modify and/ or
// redistribute the software under the terms of the CeCILL license as circulated
// by CEA, CNRS and INRIA at the following URL http://www.cecill.info/
//
// As a counterpart to the access to the source code and rights to copy, modify
// and redistribute granted by the license, users are provided only with a
// limited warranty and the software's author, the holder of the economic
// rights, and the successive licensors have only limited liability.
//
// In this respect, the user's attention is drawn to the risks associated with
// loading, using, modifying and/or developing or reproducing the software by
// the user in light of its specific status of free software, that may mean that
// it is complicated to manipulate, and that also therefore means that it is
// reserved for developers and experienced professionals having in-depth
// computer knowledge. Users are therefore encouraged to load and test the
// software's suitability as regards their requirements in conditions enabling
// the security of their systems and/or data to be ensured and, more generally,
// to use and operate it in the same conditions as regards security.
//
// The fact that you are presently reading this means that you have had
// knowledge of the CeCILL license and that you accept its terms.
// ----------------------------------------------------------------------------
// Reaction-diffusion fragment shader
// Copyright (C) 2008,2009,2010  Claude Heiland-Allen
// ----------------------------------------------------------------------------
uniform sampler2D texture; // U,V:= r,g, other channels ignored
uniform sampler2D params;  // rU,rV,f,k := r,g,b,a
uniform float dx;          // horizontal distance between texels
uniform float dy;          // vertical distance between texels
uniform float dd;          // unit of distance
uniform float dt;          // unit of time

void main(void)
{
    float center = -(4.0+4.0/sqrt(2.0));  // -1 * other weights
    float diag   = 1.0/sqrt(2.0);       // weight for diagonals
    vec2 p = gl_TexCoord[0].st;             // center coordinates
    vec2 c = texture2D(texture, p).rg;      // center value

    vec2 l                                  // compute Laplacian
        = ( texture2D(texture, p + vec2(-dx,-dy)).rg
          + texture2D(texture, p + vec2( dx,-dy)).rg
          + texture2D(texture, p + vec2(-dx, dy)).rg
          + texture2D(texture, p + vec2( dx, dy)).rg) * diag
        + texture2D(texture, p + vec2(-dx, 0.0)).rg
        + texture2D(texture, p + vec2( dx, 0.0)).rg
        + texture2D(texture, p + vec2(0.0,-dy)).rg
        + texture2D(texture, p + vec2(0.0, dy)).rg
        + c.rg * center;

    float u = c.r;           // compute some temporary
    float v = c.g;           // values which might save
    float lu = l.r;          // a few GPU cycles
    float lv = l.g;
    float uvv = u * v * v;

    vec4 q = texture2D(params, p).rgba;
    float ru = q.r;          // rate of diffusion of U
    float rv = q.g;          // rate of diffusion of V
    float f  = q.b;          // some coupling parameter
    float k  = q.a;          // another coupling parameter

    float du = ru * lu / dd - uvv + f * (1.0 - u); // Gray-Scott equation
    float dv = rv * lv / dd + uvv - (f + k) * v;   // diffusion+-reaction

    u += du * dt;
    v += dv * dt;
    gl_FragColor = vec4(clamp(u, 0.0, 1.0), clamp(v, 0.0, 1.0), 0.0, 1.0); // output new (u,v)
}
