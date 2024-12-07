/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2024 Stanford University and the Authors.           *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "openmm/internal/AssertionUtilities.h"
#include "openmm/SASAForce.h"
#include "openmm/Context.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "sfmt/SFMT.h"
#include <cmath>
#include <iostream>
#include <vector>

using namespace OpenMM;
using namespace std;

void testOneParticle() {
    System system;
    system.addParticle(1.0);
    SASAForce* sasa = new SASAForce(1.5);
    sasa->addParticle(3.5);
    system.addForce(sasa);
    vector<Vec3> positions(1);
    positions[0] = Vec3(0.5, 1.5, -2.0);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    State state = context.getState(State::Energy | State::Forces);
    ASSERT_EQUAL_TOL(1.5*4.0*M_PI*3.5*3.5, state.getPotentialEnergy(), 1e-5);
    Vec3 expected;
    ASSERT_EQUAL_VEC(expected, state.getForces()[0], 1e-5);

    // Check that updateParametersInContext() works correctly.

    sasa->setParticleParameters(0, 2.5);
    sasa->updateParametersInContext(context);
    state = context.getState(State::Energy);
    ASSERT_EQUAL_TOL(1.5*4.0*M_PI*2.5*2.5, state.getPotentialEnergy(), 1e-5);

    // Check that modifying the scale factor works correctly.

    context.setParameter(SASAForce::EnergyScale(), 3.0);
    state = context.getState(State::Energy);
    ASSERT_EQUAL_TOL(3.0*4.0*M_PI*2.5*2.5, state.getPotentialEnergy(), 1e-5);
}

void testTwoParticles() {
    double radius1 = 2.0;
    double radius2 = 4.0;
    double d = 3.5;
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    SASAForce* sasa = new SASAForce(1.3);
    sasa->addParticle(radius1);
    sasa->addParticle(radius2);
    system.addForce(sasa);
    vector<Vec3> positions(2);
    positions[0] = Vec3(0.0, 0.0, 0.0);
    positions[0] = Vec3(d, 0.0, 0.0);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    State state = context.getState(State::Energy);
    double intersectRadius = (0.5/d)*sqrt((radius1-radius2-d)*(radius2-radius1-d)*(radius1+radius2-d)*(radius1+radius2+d));
    double angle1 = asin(intersectRadius/radius1);
    double angle2 = asin(intersectRadius/radius2);
    double sin1 = sin(angle1/2);
    double sin2 = sin(angle2/2);
    double expectedEnergy = 1.3*4.0*M_PI*(radius1*radius1*(1-sin1*sin1) + radius2*radius2*(1-sin2*sin2));
    ASSERT_EQUAL_TOL(expectedEnergy, state.getPotentialEnergy(), 1e-5);
}

void testForces() {
    const int numParticles = 20;
    System system;
    SASAForce* force = new SASAForce(2.5);
    system.addForce(force);
    vector<Vec3> positions(numParticles);
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    for (int i = 0; i < numParticles; ++i) {
        system.addParticle(1.0);
        force->addParticle(4*genrand_real2(sfmt));
        positions[i] = Vec3(genrand_real2(sfmt), genrand_real2(sfmt), genrand_real2(sfmt))*10;
    }
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    
    // Compute the initial energy
    
    State state1 = context.getState(State::Energy | State::Forces);

    // Translate and rotate all the particles.  This should have no effect on the SASA.

    vector<Vec3> transformedPos(numParticles);
    double cs = cos(1.1), sn = sin(1.1);
    for (int i = 0; i < numParticles; i++) {
        Vec3 p = positions[i];
        transformedPos[i] = Vec3( cs*p[0] + sn*p[1] + 0.1,
                                 -sn*p[0] + cs*p[1] - 11.3,
                                  p[2] + 1.5);
    }
    context.setPositions(transformedPos);
    State state2 = context.getState(State::Energy | State::Forces);
    ASSERT_EQUAL_TOL(state1.getPotentialEnergy(), state2.getPotentialEnergy(), 1e-4);
    for (int i = 0; i < numParticles; i++) {
        Vec3 f = state1.getForces()[i];
        Vec3 transformedForce( cs*f[0] + sn*f[1],
                              -sn*f[0] + cs*f[1],
                                  f[2]);
        ASSERT_EQUAL_VEC(transformedForce, state2.getForces()[i], 1e-4);
    }

    // Take a small step in the direction of the energy gradient and see whether the potential energy changes by the expected amount.

    const vector<Vec3>& forces = state2.getForces();
    double norm = 0.0;
    for (int i = 0; i < (int) forces.size(); ++i)
        norm += forces[i].dot(forces[i]);
    norm = std::sqrt(norm);
    const double stepSize = 0.01;
    double step = 0.5*stepSize/norm;
    vector<Vec3> positions2(numParticles), positions3(numParticles);
    for (int i = 0; i < (int) positions.size(); ++i) {
        Vec3 p = transformedPos[i];
        Vec3 f = forces[i];
        positions2[i] = Vec3(p[0]-f[0]*step, p[1]-f[1]*step, p[2]-f[2]*step);
        positions3[i] = Vec3(p[0]+f[0]*step, p[1]+f[1]*step, p[2]+f[2]*step);
    }
    context.setPositions(positions2);
    State state3 = context.getState(State::Energy);
    context.setPositions(positions3);
    State state4 = context.getState(State::Energy);
    ASSERT_EQUAL_TOL(norm, (state3.getPotentialEnergy()-state4.getPotentialEnergy())/stepSize, 1e-3);
}

void runPlatformTests();

int main(int argc, char* argv[]) {
    try {
        initializeTests(argc, argv);
        testOneParticle();
        testTwoParticles();
        testForces();
        runPlatformTests();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
