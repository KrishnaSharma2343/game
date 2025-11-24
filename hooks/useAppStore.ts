/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/


import React, { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import { AiStage, type Slider, type TerraformConfig, TerraformTarget, ControlConfig, SoundConfig, Modulation, ModulationSource, ModulationTarget, CameraData, ViewMode, ShipConfig, ShipModulation, ShipModulationTarget, SliderSuggestion } from '../types';
import { AppContextType } from '../context/AppContext';
import { v4 as uuidv4 } from 'uuid';
import { EDITMODE } from '../config';
import {
    analyzeShaderForSliders,
    enrichSliderDetails,
    fetchSliderSuggestions,
    explainCode,
    determineModificationType,
    adjustSliders,
    createSmartSlider,
    implementCameraControls,
    modifyCode,
    fixCode,
    generateAudioModulation
} from '../services/GeminiService';

interface SessionState {
  sessionId?: string;
  shaderCode?: string;
  sliders?: Slider[];
  uniforms?: { [key:string]: number };
  cameraControlsEnabled?: boolean;
  terraformConfig?: TerraformConfig;
  controlConfig?: ControlConfig;
  soundConfig?: SoundConfig;
  source?: string;
  shipConfig?: ShipConfig;
  // New Settings
  canvasSize?: string;
  viewMode?: ViewMode;
  isHdEnabled?: boolean;
  isFpsEnabled?: boolean;
  isHudEnabled?: boolean;
  collisionThresholdRed?: number;
  collisionThresholdYellow?: number;
}

// Helpers for URL hash management
const parseHash = (): Record<string, string> => {
    const hash = window.location.hash.substring(1);
    if (!hash) return {};
    const params: Record<string, string> = {};
    hash.split('&').forEach(part => {
        const temp = part.split('=');
        if (temp.length === 2) {
            params[decodeURIComponent(temp[0])] = decodeURIComponent(temp[1]);
        }
    });
    return params;
};

const stringifyHash = (params: Record<string, string>): string => {
    return Object.entries(params)
        .map(([key, value]) => `${encodeURIComponent(key)}=${encodeURIComponent(value)}`)
        .join('&');
};

// --- Optimized Math Helpers for JS Raymarching ---
// Using typed arrays and strictly avoiding new object creation in loops.

const temp_q = new Float32Array(3);
const temp_q_rot = new Float32Array(3);

const getPlanet1Distance = (p_vec: number[] | Float32Array, uniforms: any, t: number) => {
    const scale = uniforms['slider_fractalScale'] ?? 0.37;
    const rot = uniforms['slider_fractalRotation'] ?? 1.09;
    const pulse = uniforms['slider_fractalPulseStrength'] ?? 0.0;

    // Copy p_vec to temp_q to avoid allocations
    temp_q[0] = p_vec[0];
    temp_q[1] = p_vec[1];
    temp_q[2] = p_vec[2];

    let d = -temp_q[1];
    let i = 58.0;

    while (i > 0.05) {
        const angle = rot + Math.sin(t * 1.0 + temp_q[1] * 5.0) * pulse;
        
        // Inline rotate3D_Y to avoid function call overhead and allocations
        const c = Math.cos(angle);
        const s = Math.sin(angle);
        // q_rotated = rotate3D_Y(angle, temp_q);
        temp_q_rot[0] = temp_q[0] * c + temp_q[2] * s;
        temp_q_rot[1] = temp_q[1];
        temp_q_rot[2] = -temp_q[0] * s + temp_q[2] * c;
        
        // Inline mod, fold, and abs logic
        const two_i = i + i;
        // q_mod = mod(q_rotated, i + i) -> ((v % y) + y) % y
        let qx = ((temp_q_rot[0] % two_i) + two_i) % two_i;
        let qy = ((temp_q_rot[1] % two_i) + two_i) % two_i;
        let qz = ((temp_q_rot[2] % two_i) + two_i) % two_i;

        // q_fold = q_mod - i
        qx -= i;
        qy -= i;
        qz -= i;

        // abs_vec(q_fold)
        qx = Math.abs(qx);
        qy = Math.abs(qy);
        qz = Math.abs(qz);

        // q = (i * 0.9) - abs_fold
        const i9 = i * 0.9;
        temp_q[0] = i9 - qx;
        temp_q[1] = i9 - qy;
        temp_q[2] = i9 - qz;

        d = Math.max(d, Math.min(temp_q[0], temp_q[1], temp_q[2]));
        i *= scale;
    }
    return d;
};


const defaultCanvasSize = '100%';

const defaultSoundConfig: SoundConfig = {
  enabled: true,
  masterVolume: 0.5,
  reverb: {
      enabled: true,
      mix: 0.5,
      decay: 5.0,
      tone: 2000,
  },
  drone: {
      enabled: true,
      gain: 0.4,
      filter: 100,
      pitch: 0,
  },
  atmosphere: {
      enabled: true,
      gain: 0.2,
      texture: 'grit',
  },
  melody: {
      enabled: true,
      gain: 0.3,
      density: 0.4,
      scale: "dorian",
  },
  arp: {
      enabled: true,
      gain: 0.25,
      speed: 1.0,
      octaves: 2,
      filter: 600,
      direction: 'updown', // Default to ping-pong if not modulated
  },
  rhythm: {
      enabled: true,
      gain: 0.4,
      bpm: 60,
      filter: 150,
  },
  // Updated Vangelis-style mappings based on user request
  modulations: [
      // Existing
      { id: '1', enabled: true, source: 'speed', target: 'drone.filter', amount: 0.4 },
      { id: '5', enabled: true, source: 'altitude', target: 'atmosphere.gain', amount: 0.15 },
      // Restored drone pitch modulation - UPDATED to -10% as requested
      { id: '6', enabled: true, source: 'altitude', target: 'drone.pitch', amount: -0.1 },
      
      // New requested mappings
      // "moves up or down based on our up/down heading" -> Pitch controls direction. Positive pitch (looking up) = UP, Negative = DOWN.
      { id: 'new1', enabled: true, source: 'pitch', target: 'arp.direction', amount: 1.5 }, 
      // "speed relates to our speed"
      { id: 'new2', enabled: true, source: 'speed', target: 'arp.speed', amount: 0.8 },
      // "octave range based on how much are we facing up or down"
      { id: 'new3', enabled: true, source: 'pitch', target: 'arp.octaves', amount: 1.0 },
  ]
}

// Modulation Ranges (what "100%" means for each target)
const MOD_RANGES: Record<ModulationTarget, number> = {
    'masterVolume': 1.0,
    'drone.gain': 1.0, 'drone.filter': 2000, 'drone.pitch': 24,
    'atmosphere.gain': 1.0,
    'arp.gain': 1.0, 'arp.speed': 3.0, 'arp.filter': 4000, 'arp.octaves': 3, '