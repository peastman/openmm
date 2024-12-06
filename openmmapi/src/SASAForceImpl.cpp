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

#include "openmm/OpenMMException.h"
#include "openmm/internal/SASAForceImpl.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/kernels.h"
#include <vector>

using namespace OpenMM;
using namespace std;

SASAForceImpl::SASAForceImpl(const SASAForce& owner) : owner(owner) {
    forceGroup = owner.getForceGroup();
}

void SASAForceImpl::initialize(ContextImpl& context) {
    kernel = context.getPlatform().createKernel(CalcSASAForceKernel::Name(), context);
    if (owner.getNumParticles() != context.getSystem().getNumParticles())
        throw OpenMMException("SASAForce must have exactly as many particles as the System it belongs to.");
    for (int i = 0; i < owner.getNumParticles(); i++) {
        double radius;
        owner.getParticleParameters(i, radius);
        if (radius <= 0)
            throw OpenMMException("SASAForce: particle radius must be positive");
    }
    kernel.getAs<CalcSASAForceKernel>().initialize(context.getSystem(), owner);
}

double SASAForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups&(1<<forceGroup)) != 0)
        return kernel.getAs<CalcSASAForceKernel>().execute(context, includeForces, includeEnergy);
    return 0.0;
}

map<string, double> SASAForceImpl::getDefaultParameters() {
    map<string, double> parameters;
    parameters[SASAForce::EnergyScale()] = getOwner().getDefaultEnergyScale();
    return parameters;
}

vector<string> SASAForceImpl::getKernelNames() {
    vector<string> names;
    names.push_back(CalcSASAForceKernel::Name());
    return names;
}

void SASAForceImpl::updateParametersInContext(ContextImpl& context) {
    kernel.getAs<CalcSASAForceKernel>().copyParametersToContext(context, owner);
    context.systemChanged();
}
