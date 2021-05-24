/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2011-2019 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
 * GNU Lesser General Public License for more details.                        *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public License   *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.      *
 * -------------------------------------------------------------------------- */

#include "CudaBondedUtilities.h"
#include "CudaContext.h"
#include "CudaExpressionUtilities.h"
#include "CudaKernelSources.h"
#include "openmm/OpenMMException.h"
#include "CudaNonbondedUtilities.h"
#include <iostream>

using namespace OpenMM;
using namespace std;

CudaBondedUtilities::CudaBondedUtilities(CudaContext& context) : context(context), numForceBuffers(0), maxBonds(0), allGroups(0), hasInitializedKernels(false) {
}

void CudaBondedUtilities::addInteraction(const vector<vector<int> >& atoms, const string& source, int group) {
    if (atoms.size() > 0) {
        forceAtoms.push_back(atoms);
        forceSource.push_back(source);
        forceGroup.push_back(group);
        allGroups |= 1<<group;
    }
}

string CudaBondedUtilities::addArgument(ArrayInterface& data, const string& type) {
    arguments.push_back(&data);
    argTypes.push_back(type);
    return "customArg"+context.intToString(arguments.size());
}

string CudaBondedUtilities::addEnergyParameterDerivative(const string& param) {
    // See if the parameter has already been added.
    
    int index;
    for (index = 0; index < energyParameterDerivatives.size(); index++)
        if (param == energyParameterDerivatives[index])
            break;
    if (index == energyParameterDerivatives.size())
        energyParameterDerivatives.push_back(param);
    context.addEnergyParameterDerivative(param);
    return string("energyParamDeriv")+context.intToString(index);
}

void CudaBondedUtilities::addPrefixCode(const string& source) {
    for (int i = 0; i < (int) prefixCode.size(); i++)
        if (prefixCode[i] == source)
            return;
    prefixCode.push_back(source);
}

void CudaBondedUtilities::initialize(const System& system) {
    int numForces = forceAtoms.size();
    hasInteractions = (numForces > 0);
    if (!hasInteractions)
        return;
    
    // Build the lists of atom indices.
    
    atomIndices.resize(numForces);
    for (int i = 0; i < numForces; i++) {
        int numBonds = forceAtoms[i].size();
        int numAtoms = forceAtoms[i][0].size();
        int numArrays = (numAtoms+3)/4;
        int startAtom = 0;
        atomIndices[i].resize(numArrays);
        for (int j = 0; j < numArrays; j++) {
            int width = min(numAtoms-startAtom, 4);
            int paddedWidth = (width == 3 ? 4 : width);
            vector<unsigned int> indexVec(paddedWidth*numBonds);
            for (int bond = 0; bond < numBonds; bond++) {
                for (int atom = 0; atom < width; atom++)
                    indexVec[bond*paddedWidth+atom] = forceAtoms[i][bond][startAtom+atom];
            }
            atomIndices[i][j].initialize(context, numBonds, 4*paddedWidth, "bondedIndices");
            atomIndices[i][j].upload(&indexVec[0]);
            startAtom += width;
        }
    }

    // Create the kernel.

    stringstream s;
    for (int i = 0; i < (int) prefixCode.size(); i++)
        s<<prefixCode[i];
    s<<"extern \"C\" __global__ void computeBondedForces(unsigned long long* __restrict__ forceBuffer, mixed* __restrict__ energyBuffer, const real4* __restrict__ posq, int groups, real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ";
    for (int force = 0; force < numForces; force++) {
        for (int i = 0; i < (int) atomIndices[force].size(); i++) {
            int indexWidth = atomIndices[force][i].getElementSize()/4;
            string indexType = "uint"+context.intToString(indexWidth);
            s<<", const "<<indexType<<"* __restrict__ atomIndices"<<force<<"_"<<i;
        }
    }
    for (int i = 0; i < (int) arguments.size(); i++)
        s<<", "<<argTypes[i]<<"* customArg"<<(i+1);
    if (energyParameterDerivatives.size() > 0)
        s<<", mixed* __restrict__ energyParamDerivs";
    s<<") {\n";
    s<<"mixed energy = 0;\n";
    for (int i = 0; i < energyParameterDerivatives.size(); i++)
        s<<"mixed energyParamDeriv"<<i<<" = 0;\n";
    for (int force = 0; force < numForces; force++)
        s<<createForceSource(force, forceAtoms[force].size(), forceAtoms[force][0].size(), forceGroup[force], forceSource[force]);
    s<<"energyBuffer[blockIdx.x*blockDim.x+threadIdx.x] += energy;\n";
    const vector<string>& allParamDerivNames = context.getEnergyParamDerivNames();
    int numDerivs = allParamDerivNames.size();
    for (int i = 0; i < energyParameterDerivatives.size(); i++)
        for (int index = 0; index < numDerivs; index++)
            if (allParamDerivNames[index] == energyParameterDerivatives[i])
                s<<"energyParamDerivs[(blockIdx.x*blockDim.x+threadIdx.x)*"<<numDerivs<<"+"<<index<<"] += energyParamDeriv"<<i<<";\n";
    s<<"}\n";
    map<string, string> defines;
    defines["PADDED_NUM_ATOMS"] = context.intToString(context.getPaddedNumAtoms());
    compiledProgram = context.compileProgramAsync(s.str(), defines);
    forceAtoms.clear();
    forceSource.clear();
}

string CudaBondedUtilities::createForceSource(int forceIndex, int numBonds, int numAtoms, int group, const string& computeForce) {
    maxBonds = max(maxBonds, numBonds);
    string suffix[] = {".x", ".y", ".z", ".w"};
    stringstream s;
    s<<"if ((groups&"<<(1<<group)<<") != 0)\n";
    s<<"for (unsigned int index = blockIdx.x*blockDim.x+threadIdx.x; index < "<<numBonds<<"; index += blockDim.x*gridDim.x) {\n";
    int startAtom = 0;
    for (int i = 0; i < (int) atomIndices[forceIndex].size(); i++) {
        int indexWidth = atomIndices[forceIndex][i].getElementSize()/4;
        string indexType = "uint"+context.intToString(indexWidth);
        s<<"    "<<indexType<<" atoms"<<i<<" = atomIndices"<<forceIndex<<"_"<<i<<"[index];\n";
        int atomsToLoad = min(indexWidth, numAtoms-startAtom);
        for (int j = 0; j < atomsToLoad; j++) {
            s<<"    unsigned int atom"<<(startAtom+j+1)<<" = atoms"<<i<<suffix[j]<<";\n";
            s<<"    real4 pos"<<(startAtom+j+1)<<" = posq[atom"<<(startAtom+j+1)<<"];\n";
        }
        startAtom += indexWidth;
    }
    s<<computeForce<<"\n";
    for (int i = 0; i < numAtoms; i++) {
        s<<"    atomicAdd(&forceBuffer[atom"<<(i+1)<<"], static_cast<unsigned long long>((long long) (force"<<(i+1)<<".x*0x100000000)));\n";
        s<<"    atomicAdd(&forceBuffer[atom"<<(i+1)<<"+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (force"<<(i+1)<<".y*0x100000000)));\n";
        s<<"    atomicAdd(&forceBuffer[atom"<<(i+1)<<"+PADDED_NUM_ATOMS*2], static_cast<unsigned long long>((long long) (force"<<(i+1)<<".z*0x100000000)));\n";
        s<<"    __threadfence_block();\n";
    }
    s<<"}\n";
    return s.str();
}

void CudaBondedUtilities::computeInteractions(int groups) {
    if ((groups&allGroups) == 0)
        return;
    if (!hasInitializedKernels) {
        hasInitializedKernels = true;
        ComputeProgram program = compiledProgram.get();
        kernel = program->createKernel("computeBondedForces");
        kernel->addArg(context.getForce());
        kernel->addArg(context.getEnergyBuffer());
        kernel->addArg(context.getPosq());
        for (int i = 0; i < 6; i++)
            kernel->addArg();
        for (int i = 0; i < (int) atomIndices.size(); i++)
            for (int j = 0; j < (int) atomIndices[i].size(); j++)
                kernel->addArg(atomIndices[i][j]);
        for (int i = 0; i < (int) arguments.size(); i++)
            kernel->addArg(*arguments[i]);
        if (energyParameterDerivatives.size() > 0)
            kernel->addArg(context.getEnergyParamDerivBuffer());
    }
    if (!hasInteractions)
        return;
    kernel->setArg(3, groups);
    Vec3 a, b, c;
    context.getPeriodicBoxVectors(a, b, c);
    if (context.getUseDoublePrecision()) {
        kernel->setArg(4, mm_double4(a[0], b[1], c[2], 0.0));
        kernel->setArg(5, mm_double4(1.0/a[0], 1.0/b[1], 1.0/c[2], 0.0));
        kernel->setArg(6, mm_double4(a[0], a[1], a[2], 0.0));
        kernel->setArg(7, mm_double4(b[0], b[1], b[2], 0.0));
        kernel->setArg(8, mm_double4(c[0], c[1], c[2], 0.0));
    }
    else {
        kernel->setArg(4, mm_float4((float) a[0], (float) b[1], (float) c[2], 0.0f));
        kernel->setArg(5, mm_float4(1.0f/(float) a[0], 1.0f/(float) b[1], 1.0f/(float) c[2], 0.0f));
        kernel->setArg(6, mm_float4((float) a[0], (float) a[1], (float) a[2], 0.0f));
        kernel->setArg(7, mm_float4((float) b[0], (float) b[1], (float) b[2], 0.0f));
        kernel->setArg(8, mm_float4((float) c[0], (float) c[1], (float) c[2], 0.0f));
    }
    kernel->execute(maxBonds);
}
