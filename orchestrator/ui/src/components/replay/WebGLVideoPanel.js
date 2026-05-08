/**
 * WebGLVideoPanel — GPU-accelerated image rendering with brightness/contrast.
 *
 * Drop-in replacement for <canvas> in the MCAP replay grid.
 * Uses Three.js ShaderMaterial to render ImageBitmap textures with
 * GPU-accelerated brightness/contrast/gamma adjustment.
 *
 * Handles:
 *  - Aspect-ratio-correct scaling (object-contain equivalent via mesh scale)
 *  - Y-flip in shader (ImageBitmap + UNPACK_FLIP_Y_WEBGL is unreliable)
 *  - Automatic resize to match CSS display dimensions
 */

import React, { useRef, useEffect, useCallback } from "react";
import * as THREE from "three";

const VERTEX_SHADER = `
  varying vec2 vUv;
  void main() {
    // Flip Y in shader — reliable across all browsers/GPUs
    // (UNPACK_FLIP_Y_WEBGL doesn't work consistently with ImageBitmap)
    vUv = vec2(uv.x, 1.0 - uv.y);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const FRAGMENT_SHADER = `
  uniform sampler2D uTexture;
  uniform float uBrightness;
  uniform float uContrast;
  uniform float uGamma;
  varying vec2 vUv;

  void main() {
    vec4 color = texture2D(uTexture, vUv);
    // Gamma correction
    color.rgb = pow(color.rgb, vec3(1.0 / uGamma));
    // Brightness
    color.rgb += uBrightness;
    // Contrast (centered around 0.5)
    color.rgb = (color.rgb - 0.5) * uContrast + 0.5;
    color.rgb = clamp(color.rgb, 0.0, 1.0);
    gl_FragColor = color;
  }
`;

const WebGLVideoPanel = React.forwardRef(function WebGLVideoPanel(
  { brightness = 0, contrast = 1, gamma = 1, className = "", style = {} },
  ref
) {
  const canvasRef = useRef(null);
  const glRef = useRef(null);
  const lastBitmapSize = useRef({ w: 0, h: 0 });

  // Initialize WebGL on mount
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    try {
      const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: false });
      renderer.setPixelRatio(1);
      renderer.setClearColor(0x111827); // bg-gray-900

      const scene = new THREE.Scene();
      const camera = new THREE.OrthographicCamera(-0.5, 0.5, 0.5, -0.5, 0.1, 10);
      camera.position.z = 1;

      const texture = new THREE.Texture();
      texture.minFilter = THREE.LinearFilter;
      texture.magFilter = THREE.LinearFilter;
      texture.generateMipmaps = false;
      texture.flipY = false; // We handle Y-flip in the vertex shader

      const material = new THREE.ShaderMaterial({
        uniforms: {
          uTexture: { value: texture },
          uBrightness: { value: brightness },
          uContrast: { value: contrast },
          uGamma: { value: gamma },
        },
        vertexShader: VERTEX_SHADER,
        fragmentShader: FRAGMENT_SHADER,
      });

      const geometry = new THREE.PlaneGeometry(1, 1);
      const mesh = new THREE.Mesh(geometry, material);
      scene.add(mesh);

      glRef.current = { renderer, scene, camera, material, texture, geometry, mesh };

      // Initial render (clear to background)
      const dw = canvas.clientWidth || 640;
      const dh = canvas.clientHeight || 480;
      renderer.setSize(dw, dh, false);
      renderer.render(scene, camera);
    } catch (e) {
      console.warn("[WebGL] Failed to init, falling back to 2D:", e);
      glRef.current = null;
    }

    return () => {
      const gl = glRef.current;
      if (gl) {
        gl.geometry.dispose();
        gl.material.dispose();
        gl.texture.dispose();
        gl.renderer.dispose();
        glRef.current = null;
      }
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Update shader uniforms when brightness/contrast/gamma change
  useEffect(() => {
    const gl = glRef.current;
    if (!gl) return;
    gl.material.uniforms.uBrightness.value = brightness;
    gl.material.uniforms.uContrast.value = contrast;
    gl.material.uniforms.uGamma.value = gamma;
    gl.renderer.render(gl.scene, gl.camera);
  }, [brightness, contrast, gamma]);

  /**
   * Render an ImageBitmap through the WebGL pipeline.
   * - Matches canvas buffer to CSS display size (sharp rendering)
   * - Scales mesh to maintain image aspect ratio (object-contain behavior)
   */
  const drawFrame = useCallback((bitmap) => {
    const gl = glRef.current;
    if (!gl) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    // Match canvas buffer to CSS display size
    const dw = canvas.clientWidth;
    const dh = canvas.clientHeight;
    if (dw === 0 || dh === 0) return;

    if (canvas.width !== dw || canvas.height !== dh) {
      gl.renderer.setSize(dw, dh, false);
    }

    // Scale mesh for aspect-ratio-correct display (object-contain equivalent)
    if (bitmap.width !== lastBitmapSize.current.w ||
        bitmap.height !== lastBitmapSize.current.h ||
        canvas.width !== dw || canvas.height !== dh) {
      const imgAspect = bitmap.width / bitmap.height;
      const viewAspect = dw / dh;

      if (imgAspect > viewAspect) {
        // Image wider than viewport: fill width, letterbox top/bottom
        gl.mesh.scale.set(1, viewAspect / imgAspect, 1);
      } else {
        // Image taller than viewport: fill height, pillarbox left/right
        gl.mesh.scale.set(imgAspect / viewAspect, 1, 1);
      }

      lastBitmapSize.current = { w: bitmap.width, h: bitmap.height };
    }

    gl.texture.image = bitmap;
    gl.texture.needsUpdate = true;
    gl.renderer.render(gl.scene, gl.camera);
  }, []);

  // Expose drawFrame via imperative handle
  React.useImperativeHandle(
    ref,
    () => ({
      drawFrame,
      getCanvas: () => canvasRef.current,
    }),
    [drawFrame]
  );

  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{ imageRendering: "auto", ...style }}
    />
  );
});

export { WebGLVideoPanel };
export default WebGLVideoPanel;
