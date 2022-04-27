
/* Portions copyright (c) 2022 Stanford University and Simbios.
 * Contributors: Pande Group
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

#ifndef __ReferenceShakeAlgorithm_H__
#define __ReferenceShakeAlgorithm_H__

#include "ReferenceConstraintAlgorithm.h"
#include <utility>
#include <vector>
#include <set>

namespace OpenMM {

/**
 * This class uses the SHAKE algorithm to enforce constraints on small clusters of atoms.
 * Each cluster consists of one central atom and up to three other atoms with the same
 * mass, whose distance to the central atom is the same.  Typically this is used to
 * constrain the distance between a heavy atom and the hydrogens bonded to it.
 */
class OPENMM_EXPORT ReferenceShakeAlgorithm : public ReferenceConstraintAlgorithm {
protected:
    int maxIterations;
    std::vector<std::vector<int> > clusterAtoms;
    std::vector<double> clusterDistance;

    virtual void applyConstraints(std::vector<OpenMM::Vec3>& atomCoordinates, std::vector<OpenMM::Vec3>& atomCoordinatesP,
                                  std::vector<double>& inverseMasses, bool constrainingVelocities, double tolerance);

    void applyToCluster(int cluster, std::vector<OpenMM::Vec3>& atomCoordinates, std::vector<OpenMM::Vec3>& atomCoordinatesP,
                        std::vector<double>& inverseMasses, bool constrainingVelocities, double tolerance);

public:
    /**
     * Create a ReferenceShakeAlgorithm object.
     * 
     * @param clusterAtoms     each element is the list of atom in a single cluster.  The first element of each
     *                         cluster is the central atom.
     * @param clusterDistance  the distance between constrained atoms for each cluster
     */
    ReferenceShakeAlgorithm(const std::vector<std::vector<int> >& clusterAtoms, const std::vector<double>& clusterDistance);
    ReferenceShakeAlgorithm(const ReferenceShakeAlgorithm& shake);

    /**
     * Get the maximum number of iterations to perform.
     */
    int getMaximumNumberOfIterations() const;

    /**
     * Set the maximum number of iterations to perform.
     */
    void setMaximumNumberOfIterations(int maximumNumberOfIterations);

    /**
     * Apply the constraint algorithm.
     * 
     * @param atomCoordinates  the original atom coordinates
     * @param atomCoordinatesP the new atom coordinates
     * @param inverseMasses    1/mass
     * @param tolerance        the constraint tolerance
     */
    void apply(std::vector<OpenMM::Vec3>& atomCoordinates,
                       std::vector<OpenMM::Vec3>& atomCoordinatesP, std::vector<double>& inverseMasses, double tolerance);

    /**
     * Apply the constraint algorithm to velocities.
     * 
     * @param atomCoordinates  the atom coordinates
     * @param atomCoordinatesP the velocities to modify
     * @param inverseMasses    1/mass
     * @param tolerance        the constraint tolerance
     */
    void applyToVelocities(std::vector<OpenMM::Vec3>& atomCoordinates,
                     std::vector<OpenMM::Vec3>& velocities, std::vector<double>& inverseMasses, double tolerance);
};

} // namespace OpenMM

#endif // __ReferenceShakeAlgorithm_H__
