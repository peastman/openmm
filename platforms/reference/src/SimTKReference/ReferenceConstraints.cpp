/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2013-2022 Stanford University and the Authors.      *
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

#include "ReferenceConstraints.h"
#include "ReferenceCCMAAlgorithm.h"
#include "ReferenceSETTLEAlgorithm.h"
#include "ReferenceShakeAlgorithm.h"
#include "openmm/HarmonicAngleForce.h"
#include "openmm/OpenMMException.h"
#include <map>
#include <utility>
#include <vector>

using namespace OpenMM;
using namespace std;

struct ReferenceConstraints::ShakeCluster {
    int centralID;
    int peripheralID[3];
    int size;
    bool valid;
    double distance;
    double centralInvMass, peripheralInvMass;
    ShakeCluster() : valid(true) {
    }
    ShakeCluster(int centralID, double invMass) : centralID(centralID), centralInvMass(invMass), size(0), valid(true) {
    }
    void addAtom(int id, double dist, double invMass) {
        if (size == 3 || (size > 0 && abs(dist-distance)/distance > 1e-8) || (size > 0 && abs(invMass-peripheralInvMass)/peripheralInvMass > 1e-8))
            valid = false;
        else {
            peripheralID[size++] = id;
            distance = dist;
            peripheralInvMass = invMass;
        }
    }
    void markInvalid(map<int, ShakeCluster>& allClusters, vector<bool>& invalidForShake)
    {
        valid = false;
        invalidForShake[centralID] = true;
        for (int i = 0; i < size; i++) {
            invalidForShake[peripheralID[i]] = true;
            map<int, ShakeCluster>::iterator otherCluster = allClusters.find(peripheralID[i]);
            if (otherCluster != allClusters.end() && otherCluster->second.valid)
                otherCluster->second.markInvalid(allClusters, invalidForShake);
        }
    }
};

ReferenceConstraints::ReferenceConstraints(const System& system) : ccma(NULL), settle(NULL), shake(NULL) {
    int numParticles = system.getNumParticles();
    vector<double> masses(numParticles);
    for (int i = 0; i < numParticles; ++i)
        masses[i] = system.getParticleMass(i);

    // Record the set of constraints and how many constraints each atom is involved in.

    vector<int> atom1;
    vector<int> atom2;
    vector<double> distance;
    vector<int> constraintCount(numParticles, 0);
    for (int i = 0; i < system.getNumConstraints(); i++) {
        int p1, p2;
        double d;
        system.getConstraintParameters(i, p1, p2, d);
        if (masses[p1] != 0 || masses[p2] != 0) {
            atom1.push_back(p1);
            atom2.push_back(p2);
            distance.push_back(d);
            constraintCount[p1]++;
            constraintCount[p2]++;
        }
    }

    // Identify clusters of three atoms that can be treated with SETTLE.  First, for every
    // atom that might be part of such a cluster, make a list of the two other atoms it is
    // connected to.

    vector<map<int, float> > settleConstraints(numParticles);
    for (int i = 0; i < (int)atom1.size(); i++) {
        if (constraintCount[atom1[i]] == 2 && constraintCount[atom2[i]] == 2) {
            settleConstraints[atom1[i]][atom2[i]] = (float) distance[i];
            settleConstraints[atom2[i]][atom1[i]] = (float) distance[i];
        }
    }

    // Now remove the ones that don't actually form closed loops of three atoms.

    vector<int> settleClusters;
    for (int i = 0; i < (int)settleConstraints.size(); i++) {
        if (settleConstraints[i].size() == 2) {
            int partner1 = settleConstraints[i].begin()->first;
            int partner2 = (++settleConstraints[i].begin())->first;
            if (settleConstraints[partner1].size() != 2 || settleConstraints[partner2].size() != 2 ||
                    settleConstraints[partner1].find(partner2) == settleConstraints[partner1].end())
                settleConstraints[i].clear();
            else if (i < partner1 && i < partner2)
                settleClusters.push_back(i);
        }
        else
            settleConstraints[i].clear();
    }

    // Record the SETTLE clusters.

    vector<bool> isSettleAtom(numParticles, false);
    if (settleClusters.size() > 0) {
        vector<int> atom1;
        vector<int> atom2;
        vector<int> atom3;
        vector<double> distance1;
        vector<double> distance2;
        for (int i = 0; i < settleClusters.size(); i++) {
            int p1 = settleClusters[i];
            int p2 = settleConstraints[p1].begin()->first;
            int p3 = (++settleConstraints[p1].begin())->first;
            float dist12 = settleConstraints[p1].find(p2)->second;
            float dist13 = settleConstraints[p1].find(p3)->second;
            float dist23 = settleConstraints[p2].find(p3)->second;
            if (dist12 == dist13) {
                // p1 is the central atom
                atom1.push_back(p1);
                atom2.push_back(p2);
                atom3.push_back(p3);
                distance1.push_back(dist12);
                distance2.push_back(dist23);
            }
            else if (dist12 == dist23) {
                // p2 is the central atom
                atom1.push_back(p2);
                atom2.push_back(p1);
                atom3.push_back(p3);
                distance1.push_back(dist12);
                distance2.push_back(dist13);
            }
            else if (dist13 == dist23) {
                // p3 is the central atom
                atom1.push_back(p3);
                atom2.push_back(p1);
                atom3.push_back(p2);
                distance1.push_back(dist13);
                distance2.push_back(dist12);
            }
            else
                continue; // We can't handle this with SETTLE
            isSettleAtom[p1] = true;
            isSettleAtom[p2] = true;
            isSettleAtom[p3] = true;
        }
        if (atom1.size() > 0)
            settle = new ReferenceSETTLEAlgorithm(atom1, atom2, atom3, distance1, distance2, masses);
    }

    // Find clusters consisting of a central atom with up to three peripheral atoms.

    map<int, ShakeCluster> clusters;
    vector<bool> invalidForShake(numParticles, false);
    for (int i = 0; i < atom1.size(); i++) {
        if (isSettleAtom[atom1[i]])
            continue; // This is being taken care of with SETTLE.

        // Determine which is the central atom.

        bool firstIsCentral;
        if (constraintCount[atom1[i]] > 1)
            firstIsCentral = true;
        else if (constraintCount[atom2[i]] > 1)
            firstIsCentral = false;
        else if (atom1[i] < atom2[i])
            firstIsCentral = true;
        else
            firstIsCentral = false;
        int centralID, peripheralID;
        if (firstIsCentral) {
            centralID = atom1[i];
            peripheralID = atom2[i];
        }
        else {
            centralID = atom2[i];
            peripheralID = atom1[i];
        }

        // Add it to the cluster.

        if (clusters.find(centralID) == clusters.end()) {
            clusters[centralID] = ShakeCluster(centralID, 1.0/system.getParticleMass(centralID));
        }
        ShakeCluster& cluster = clusters[centralID];
        cluster.addAtom(peripheralID, distance[i], 1.0/system.getParticleMass(peripheralID));
        if (constraintCount[peripheralID] != 1 || invalidForShake[atom1[i]] || invalidForShake[atom2[i]]) {
            cluster.markInvalid(clusters, invalidForShake);
            map<int, ShakeCluster>::iterator otherCluster = clusters.find(peripheralID);
            if (otherCluster != clusters.end() && otherCluster->second.valid)
                otherCluster->second.markInvalid(clusters, invalidForShake);
        }
    }
    vector<vector<int> > clusterAtoms;
    vector<double> clusterDistance;
    for (map<int, ShakeCluster>::iterator iter = clusters.begin(); iter != clusters.end(); ++iter) {
        ShakeCluster& cluster = iter->second;
        if (cluster.valid) {
            cluster.valid = !invalidForShake[cluster.centralID] && cluster.size == constraintCount[cluster.centralID];
            for (int i = 0; i < cluster.size; i++)
                if (invalidForShake[cluster.peripheralID[i]])
                    cluster.valid = false;
            if (cluster.valid) {
                clusterAtoms.push_back(vector<int>());
                clusterAtoms.back().push_back(cluster.centralID);
                isSettleAtom[cluster.centralID] = true;
                for (int i = 0; i < cluster.size; i++) {
                    clusterAtoms.back().push_back(cluster.peripheralID[i]);
                    isSettleAtom[cluster.peripheralID[i]] = true;
                }
                clusterDistance.push_back(cluster.distance);
            }
        }
    }
    if (clusterAtoms.size() > 0)
        shake = new ReferenceShakeAlgorithm(clusterAtoms, clusterDistance);

    // All other constraints are handled with CCMA.

    vector<int> ccmaConstraints;
    for (unsigned i = 0; i < atom1.size(); i++)
        if (!isSettleAtom[atom1[i]])
            ccmaConstraints.push_back(i);
    int numCCMA = ccmaConstraints.size();
    if (numCCMA > 0) {
        // Record particles and distances for CCMA.
        
        vector<pair<int, int> > ccmaIndices(numCCMA);
        vector<double> ccmaDistance(numCCMA);
        for (int i = 0; i < numCCMA; i++) {
            int index = ccmaConstraints[i];
            ccmaIndices[i] = make_pair(atom1[index], atom2[index]);
            ccmaDistance[i] = distance[index];
        }

        // Look up angles for CCMA.
        
        vector<ReferenceCCMAAlgorithm::AngleInfo> angles;
        for (int i = 0; i < system.getNumForces(); i++) {
            const HarmonicAngleForce* force = dynamic_cast<const HarmonicAngleForce*>(&system.getForce(i));
            if (force != NULL) {
                for (int j = 0; j < force->getNumAngles(); j++) {
                    int atom1, atom2, atom3;
                    double angle, k;
                    force->getAngleParameters(j, atom1, atom2, atom3, angle, k);
                    angles.push_back(ReferenceCCMAAlgorithm::AngleInfo(atom1, atom2, atom3, angle));
                }
            }
        }
        
        // Create the CCMA object.
        
        ccma = new ReferenceCCMAAlgorithm(numParticles, numCCMA, ccmaIndices, ccmaDistance, masses, angles, 0.02);
    }
}

ReferenceConstraints::~ReferenceConstraints() {
    if (ccma != NULL)
        delete ccma;
    if (settle != NULL)
        delete settle;
    if (shake != NULL)
        delete shake;
}

void ReferenceConstraints::apply(vector<OpenMM::Vec3>& atomCoordinates, vector<OpenMM::Vec3>& atomCoordinatesP, vector<double>& inverseMasses, double tolerance) {
    if (ccma != NULL)
        ccma->apply(atomCoordinates, atomCoordinatesP, inverseMasses, tolerance);
    if (settle != NULL)
        settle->apply(atomCoordinates, atomCoordinatesP, inverseMasses, tolerance);
    if (shake != NULL)
        shake->apply(atomCoordinates, atomCoordinatesP, inverseMasses, tolerance);
}

void ReferenceConstraints::applyToVelocities(vector<OpenMM::Vec3>& atomCoordinates, vector<OpenMM::Vec3>& velocities, vector<double>& inverseMasses, double tolerance) {
    if (ccma != NULL)
        ccma->applyToVelocities(atomCoordinates, velocities, inverseMasses, tolerance);
    if (settle != NULL)
        settle->applyToVelocities(atomCoordinates, velocities, inverseMasses, tolerance);
    if (shake != NULL)
        shake->applyToVelocities(atomCoordinates, velocities, inverseMasses, tolerance);
}
