#!/usr/bin/env python3
"""Quick test to see if Genesis can load our G1 model."""

import time


def test_genesis():
    """Test Genesis with G1 MJCF model."""
    print("=" * 50)
    print("Genesis G1 Test")
    print("=" * 50)

    # Import Genesis
    print("\n1. Importing Genesis...")
    start = time.time()
    import genesis as gs

    print(f"   Import time: {time.time() - start:.2f}s")

    # Initialize with CPU backend (Metal also available on Mac)
    print("\n2. Initializing Genesis (CPU backend)...")
    start = time.time()
    gs.init(backend=gs.cpu)
    print(f"   Init time: {time.time() - start:.2f}s")

    # Create scene
    print("\n3. Creating scene...")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,  # 100Hz simulation
            substeps=4,
        ),
        show_viewer=False,  # No GUI for this test
    )

    # Add ground plane
    print("   Adding ground plane...")
    plane = scene.add_entity(gs.morphs.Plane())

    # Load G1 robot from MJCF
    print("\n4. Loading G1 MJCF model...")
    start = time.time()
    try:
        robot = scene.add_entity(
            gs.morphs.MJCF(
                file="unitree_rl_gym/resources/robots/g1_description/g1_12dof.xml",
                pos=(0, 0, 1.0),  # Start 1m above ground
            ),
        )
        print(f"   Load time: {time.time() - start:.2f}s")
        print(f"   Robot loaded successfully!")
    except Exception as e:
        print(f"   Failed to load MJCF: {e}")
        print("\n   Trying URDF instead...")
        # Try URDF as fallback
        try:
            robot = scene.add_entity(
                gs.morphs.URDF(
                    file="unitree_rl_gym/resources/robots/g1_description/urdf/g1_12dof.urdf",
                    pos=(0, 0, 1.0),
                ),
            )
            print(f"   URDF loaded successfully!")
        except Exception as e2:
            print(f"   URDF also failed: {e2}")
            return False

    # Build scene
    print("\n5. Building scene...")
    start = time.time()
    scene.build()
    print(f"   Build time: {time.time() - start:.2f}s")

    # Run simulation benchmark
    print("\n6. Running simulation benchmark...")
    n_steps = 1000
    start = time.time()
    for _ in range(n_steps):
        scene.step()
    elapsed = time.time() - start

    sim_time = n_steps * 0.01  # 10 seconds of sim time
    rtf = sim_time / elapsed  # Real-time factor

    print(f"   Steps: {n_steps}")
    print(f"   Sim time: {sim_time:.1f}s")
    print(f"   Wall time: {elapsed:.3f}s")
    print(f"   Real-time factor: {rtf:.1f}x")
    print(f"   FPS: {n_steps / elapsed:.0f}")

    print("\n" + "=" * 50)
    print("Genesis test PASSED!")
    print("=" * 50)
    return True


if __name__ == "__main__":
    success = test_genesis()
    exit(0 if success else 1)
