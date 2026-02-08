"use client";

import React, { useRef, useEffect, Suspense } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Grid, Environment, PerspectiveCamera } from "@react-three/drei";
import * as THREE from "three";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/store/useAppStore";
import type { TrajectoryFrame } from "@/types/trajectory";
import { ATTEMPT_COLORS } from "@/types/trajectory";

interface ThreeCanvasProps {
  className?: string;
}

/**
 * Simple robot visualization (placeholder for MuJoCo bodies)
 */
function Robot({ frame }: { frame: TrajectoryFrame | null }) {
  const meshRef = useRef<THREE.Mesh>(null);

  if (!frame) return null;

  // Extract position from frame
  const position = frame.position || [0, 0.5, 0];
  const quaternion = frame.quaternion || [0, 0, 0, 1];

  return (
    <group position={[position[0], position[1], -position[2]]}>
      {/* Body */}
      <mesh ref={meshRef}>
        <capsuleGeometry args={[0.15, 0.6, 4, 16]} />
        <meshStandardMaterial color="#4CAF50" roughness={0.4} metalness={0.6} />
      </mesh>

      {/* Head */}
      <mesh position={[0, 0.5, 0]}>
        <sphereGeometry args={[0.12, 16, 16]} />
        <meshStandardMaterial color="#4CAF50" roughness={0.4} metalness={0.6} />
      </mesh>

      {/* Direction indicator */}
      <mesh position={[0.15, 0.5, 0]} rotation={[0, 0, -Math.PI / 2]}>
        <coneGeometry args={[0.05, 0.1, 8]} />
        <meshStandardMaterial color="#fff" emissive="#fff" emissiveIntensity={0.5} />
      </mesh>
    </group>
  );
}

/**
 * Path trail visualization
 */
function PathTrail({ frames, currentFrame }: { frames: TrajectoryFrame[]; currentFrame: number }) {
  const lineRef = useRef<THREE.Line>(null);

  // Build path points up to current frame
  const points = React.useMemo(() => {
    if (!frames.length) return [];

    const pts: THREE.Vector3[] = [];
    const maxFrame = Math.min(currentFrame + 1, frames.length);

    for (let i = 0; i < maxFrame; i++) {
      const frame = frames[i];
      const pos = frame.position || [0, 0.1, 0];
      pts.push(new THREE.Vector3(pos[0], 0.05, -pos[2]));
    }

    return pts;
  }, [frames, currentFrame]);

  // Create geometry
  const geometry = React.useMemo(() => {
    if (points.length < 2) return null;
    return new THREE.BufferGeometry().setFromPoints(points);
  }, [points]);

  if (!geometry) return null;

  return (
    <primitive object={new THREE.Line(geometry, new THREE.LineBasicMaterial({ color: ATTEMPT_COLORS[0] }))} ref={lineRef} />
  );
}

/**
 * Forbidden zone visualization
 */
function ForbiddenZone({ position, size }: { position: [number, number, number]; size: [number, number, number] }) {
  return (
    <mesh position={position}>
      <boxGeometry args={size} />
      <meshStandardMaterial
        color="#f44336"
        transparent
        opacity={0.15}
        side={THREE.DoubleSide}
      />
      {/* Wireframe */}
      <lineSegments>
        <edgesGeometry args={[new THREE.BoxGeometry(...size)]} />
        <lineBasicMaterial color="#f44336" linewidth={2} />
      </lineSegments>
    </mesh>
  );
}

/**
 * Goal marker visualization
 */
function GoalMarker({ position }: { position: [number, number, number] }) {
  return (
    <group position={position}>
      {/* Circle on ground */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.01, 0]}>
        <ringGeometry args={[0.3, 0.4, 32]} />
        <meshBasicMaterial color="#4CAF50" side={THREE.DoubleSide} />
      </mesh>

      {/* Vertical beam */}
      <mesh position={[0, 1, 0]}>
        <cylinderGeometry args={[0.02, 0.02, 2, 8]} />
        <meshBasicMaterial color="#4CAF50" transparent opacity={0.5} />
      </mesh>

      {/* Flag */}
      <mesh position={[0.15, 1.8, 0]}>
        <planeGeometry args={[0.3, 0.2]} />
        <meshBasicMaterial color="#4CAF50" side={THREE.DoubleSide} />
      </mesh>
    </group>
  );
}

/**
 * Barrel obstacle
 */
function Barrel({ position }: { position: [number, number, number] }) {
  return (
    <mesh position={position}>
      <cylinderGeometry args={[0.2, 0.2, 0.5, 16]} />
      <meshStandardMaterial color="#FF9800" roughness={0.6} metalness={0.2} />
    </mesh>
  );
}

/**
 * Camera controller with follow mode
 */
function CameraController({ followPosition }: { followPosition: [number, number, number] | null }) {
  const { camera } = useThree();
  const controlsRef = useRef<any>(null);

  useFrame(() => {
    if (followPosition && controlsRef.current) {
      // Smoothly update target
      const target = controlsRef.current.target;
      target.lerp(new THREE.Vector3(followPosition[0], 0.8, -followPosition[2]), 0.1);
    }
  });

  return (
    <OrbitControls
      ref={controlsRef}
      target={[0, 0.5, 0]}
      enableDamping
      dampingFactor={0.05}
      minDistance={1}
      maxDistance={20}
      maxPolarAngle={Math.PI / 2}
    />
  );
}

/**
 * Scene content
 */
function SceneContent() {
  const { trajectory, playback } = useAppStore();

  const currentFrame = trajectory?.frames?.[playback.currentFrame] || null;
  const frames = trajectory?.frames || [];

  // Example scene data (would come from trajectory metadata)
  const goalPosition: [number, number, number] = [2, 0, 0];
  const barrelPositions: [number, number, number][] = [
    [1, 0.25, 0.5],
    [1.5, 0.25, -0.3],
    [0.8, 0.25, -0.2],
  ];
  const forbiddenZone: { position: [number, number, number]; size: [number, number, number] } = {
    position: [1, 0.5, 0],
    size: [1.5, 1, 1.5],
  };

  const followPosition = currentFrame?.position as [number, number, number] | null;

  return (
    <>
      {/* Camera */}
      <PerspectiveCamera makeDefault position={[3, 3, 3]} fov={45} />
      <CameraController followPosition={followPosition} />

      {/* Lighting */}
      <ambientLight intensity={0.4} />
      <directionalLight position={[5, 10, 5]} intensity={1} castShadow />
      <directionalLight position={[-5, 5, -5]} intensity={0.3} />

      {/* Environment */}
      <Environment preset="city" background={false} />

      {/* Ground */}
      <Grid
        args={[20, 20]}
        cellSize={0.5}
        cellThickness={0.5}
        cellColor="#333"
        sectionSize={2}
        sectionThickness={1}
        sectionColor="#444"
        fadeDistance={30}
        position={[0, 0, 0]}
      />

      {/* Ground plane for shadows */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.01, 0]} receiveShadow>
        <planeGeometry args={[20, 20]} />
        <meshStandardMaterial color="#1a1a1a" />
      </mesh>

      {/* Scene elements */}
      <GoalMarker position={goalPosition} />
      <ForbiddenZone {...forbiddenZone} />

      {barrelPositions.map((pos, i) => (
        <Barrel key={i} position={pos} />
      ))}

      {/* Robot and trail */}
      <Robot frame={currentFrame} />
      <PathTrail frames={frames} currentFrame={playback.currentFrame} />
    </>
  );
}

/**
 * Loading placeholder
 */
function LoadingPlaceholder() {
  return (
    <div className="absolute inset-0 flex items-center justify-center bg-[var(--color-bg-base)]">
      <div className="text-center">
        <div className="w-12 h-12 border-4 border-white/20 border-t-[var(--color-accent-primary)] rounded-full animate-spin mx-auto mb-4" />
        <p className="text-white/60">Loading 3D scene...</p>
      </div>
    </div>
  );
}

/**
 * Empty state
 */
function EmptyState() {
  return (
    <div className="absolute inset-0 flex items-center justify-center bg-[var(--color-bg-base)]">
      <div className="text-center max-w-md px-8">
        <div className="text-6xl mb-4">🤖</div>
        <h2 className="text-xl font-semibold text-white mb-2">
          No Experiment Loaded
        </h2>
        <p className="text-white/60 mb-6">
          Select an experiment from the dropdown above to view the robot&apos;s
          trajectory and AI decision-making process.
        </p>
        <div className="flex flex-wrap justify-center gap-2 text-sm">
          <kbd className="px-2 py-1 rounded bg-white/10 text-white/70">Space</kbd>
          <span className="text-white/40">Play/Pause</span>
          <kbd className="px-2 py-1 rounded bg-white/10 text-white/70">←→</kbd>
          <span className="text-white/40">Step</span>
          <kbd className="px-2 py-1 rounded bg-white/10 text-white/70">G</kbd>
          <span className="text-white/40">Chat</span>
        </div>
      </div>
    </div>
  );
}

/**
 * Three.js canvas wrapper component
 */
export function ThreeCanvas({ className }: ThreeCanvasProps) {
  const { trajectory, isLoading } = useAppStore();

  if (isLoading) {
    return <LoadingPlaceholder />;
  }

  if (!trajectory) {
    return <EmptyState />;
  }

  return (
    <div
      className={cn(
        "absolute inset-0",
        "top-[var(--header-height)] bottom-[var(--playback-bar-height)]",
        className
      )}
    >
      <Canvas
        shadows
        gl={{
          antialias: true,
          alpha: true,
          logarithmicDepthBuffer: true,
        }}
        style={{ background: "#121212" }}
      >
        <Suspense fallback={null}>
          <SceneContent />
        </Suspense>
      </Canvas>
    </div>
  );
}

export default ThreeCanvas;
