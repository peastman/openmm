
/* Portions copyright (c) 2022 Stanford University and Simbios.
 * Contributors: Peter Eastman, Pande Group
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "ReferenceShakeAlgorithm.h"
#include "openmm/Vec3.h"

using namespace OpenMM;
using namespace std;

ReferenceShakeAlgorithm::ReferenceShakeAlgorithm(const vector<vector<int> >& clusterAtoms, const vector<double>& clusterDistance) :
        clusterAtoms(clusterAtoms), clusterDistance(clusterDistance), maxIterations(15) {
}

ReferenceShakeAlgorithm::ReferenceShakeAlgorithm(const ReferenceShakeAlgorithm& shake) : clusterAtoms(shake.clusterAtoms),
        clusterDistance(shake.clusterDistance), maxIterations(shake.maxIterations) {
}

int ReferenceShakeAlgorithm::getMaximumNumberOfIterations() const {
    return maxIterations;
}

void ReferenceShakeAlgorithm::setMaximumNumberOfIterations(int maximumNumberOfIterations) {
    maxIterations = maximumNumberOfIterations;
}

void ReferenceShakeAlgorithm::apply(vector<Vec3>& atomCoordinates,
                                         vector<Vec3>& atomCoordinatesP,
                                         vector<double>& inverseMasses, double tolerance) {
    applyConstraints(atomCoordinates, atomCoordinatesP, inverseMasses, false, tolerance);
}

void ReferenceShakeAlgorithm::applyToVelocities(std::vector<OpenMM::Vec3>& atomCoordinates,
               std::vector<OpenMM::Vec3>& velocities, std::vector<double>& inverseMasses, double tolerance) {
    applyConstraints(atomCoordinates, velocities, inverseMasses, true, tolerance);
}

void ReferenceShakeAlgorithm::applyConstraints(vector<Vec3>& atomCoordinates,
                                         vector<Vec3>& atomCoordinatesP,
                                         vector<double>& inverseMasses, bool constrainingVelocities, double tolerance) {
    for (int cluster = 0; cluster < clusterAtoms.size(); cluster++)
        applyToCluster(cluster, atomCoordinates, atomCoordinatesP, inverseMasses, constrainingVelocities, tolerance);
}

void ReferenceShakeAlgorithm::applyToCluster(int cluster, vector<Vec3>& atomCoordinates,
                                         vector<Vec3>& atomCoordinatesP,
                                         vector<double>& inverseMasses, bool constrainingVelocities, double tolerance) {
    const vector<int>& atoms = clusterAtoms[cluster];
    double d2 = clusterDistance[cluster]*clusterDistance[cluster];
    double invMassCentral = inverseMasses[atoms[0]];
    bool converged = false;
    for (int iteration = 0; iteration < maxIterations && !converged; iteration++) {
        converged = true;
        for (int i = 1; i < atoms.size(); i++) {
            Vec3 rij = atomCoordinates[atoms[0]]-atomCoordinates[atoms[i]];
            Vec3 rpij = (atomCoordinatesP[atoms[0]]-atomCoordinatesP[atoms[i]])-rij;
            double rijsq = rij.dot(rij);
            double rrpr = rij.dot(rpij);
            double avgMass = 0.5/(invMassCentral+inverseMasses[atoms[i]]);
            if (constrainingVelocities) {
                double delta = -2*avgMass*rrpr/rijsq;
                Vec3 dr = rij*delta;
                atomCoordinatesP[atoms[0]] += dr*invMassCentral;
                atomCoordinatesP[atoms[i]] -= dr*inverseMasses[atoms[i]];
                if (fabs(delta) > tolerance)
                    converged = false;
            }
            else {
                double rpijsq = rpij.dot(rpij);
                double ld = d2-rijsq;
                double diff = fabs(ld-2*rrpr-rpijsq) / (d2*tolerance);
                if (diff >= 1.0) {
                    double acor  = (ld-2*rrpr-rpijsq)*avgMass / (rrpr+rijsq);
                    Vec3 dr = rij*acor;
                    atomCoordinatesP[atoms[0]] += dr*invMassCentral;
                    atomCoordinatesP[atoms[i]] -= dr*inverseMasses[atoms[i]];
                    converged = false;
                }
            }
        }
    }
}
