#ifndef OPENMM_SASAFORCE_H_
#define OPENMM_SASAFORCE_H_

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

#include "Force.h"
#include <vector>
#include "internal/windowsExport.h"

namespace OpenMM {

/**
 * This class implements an energy that is proportional to the solvent accessible
 * surface area (SASA) of the system.  Potentials of this form are commonly used
 * in implicit solvent models.
 * 
 * To use this class, create a SASAForce object, then call addParticle() once for each particle in the
 * System to define its radius.  The number of particles for which you define SASA parameters must
 * be exactly equal to the number of particles in the System, or else an exception will be thrown when you
 * try to create a Context.  After a particle has been added, you can modify its radius by calling
 * setParticleParameters().  This will have no effect on Contexts that already exist unless you
 * call updateParametersInContext().
 * 
 * When you create a SASAForce, you specify the scale factor by which the surface
 * area is multiplied to get the potential energy.  The value provided to the constructor
 * is a default value that will apply to newly created Contexts.  You can change
 * the scale factor for an existing Context by calling setParameter() on it:
 * 
 * \verbatim embed:rst:leading-asterisk
 * .. code-block:: cpp
 *
 *    context.setParameter(SASAForce::EnergyScale(), scaleFactor);
 * \endverbatim
 */

class OPENMM_EXPORT SASAForce : public Force {
public:
    /**
     * This is the name of the parameter which stores the current energy scale (in kJ/mol/nm^2).
     */
    static const std::string& EnergyScale() {
        static const std::string key = "SASAEnergyScale";
        return key;
    }
    /**
     * Create a SASAForce.
     * 
     * @param defaultEnergyScale     the default scale factor for the energy (in kJ/mol/nm^2).
     */
    SASAForce(double defaultEnergyScale);
    /**
     * Get the default energy scale by which the surface area is multiplied, measured in kJ/mol/nm^2.
     */
    double getDefaultEnergyScale() const;
    /**
     * Set the default energy scale by which the surface area is multiplied, measured in kJ/mol/nm^2.
     * This will apply to newly created Contexts.  To change the energy scale for an existing Context,
     * call setParameter() on it.
     */
    double setDefaultEnergyScale(double defaultEnergyScale);
    /**
     * Get the number of particles in the system.
     */
    int getNumParticles() const {
        return particles.size();
    }
    /**
     * Add a particle to the force.  This should be called once for each particle
     * in the System.  When it is called for the i'th time, it specifies the radius
     * for the i'th particle.
     *
     * @param radius         the radius of the particle, measured in nm
     * @return the index of the particle that was added
     */
    int addParticle(double radius);
    /**
     * Get the parameters for a particle.
     *
     * @param index               the index of the particle for which to get parameters
     * @param[out] radius         the radius of the particle, measured in nm
     */
    void getParticleParameters(int index, double& radius) const;
    /**
     * Set the parameters for a particle.
     *
     * @param index          the index of the particle for which to set parameters
     * @param radius         the radius of the particle, measured in nm
     */
    void setParticleParameters(int index, double radius);
    /**
     * Update the particle parameters in a Context to match those stored in this Force object.  This method
     * provides an efficient method to update certain parameters in an existing Context without needing to
     * reinitialize it.  Simply call setParticleParameters() to modify this object's parameters, then call
     * updateParametersInContext() to copy them over to the Context.
     *
     * This method cannot be used to add new particles, only to change the parameters of existing ones.
     */
    void updateParametersInContext(Context& context);
    /**
     * Returns whether or not this force makes use of periodic boundary
     * conditions.
     *
     * @returns true if force uses PBC and false otherwise
     */
    bool usesPeriodicBoundaryConditions() const {
        return false;
    }
protected:
    ForceImpl* createImpl() const;
private:
    class ParticleInfo;
    double defaultEnergyScale;
    std::vector<ParticleInfo> particles;
};

/**
 * This is an internal class used to record information about a particle.
 * @private
 */
class SASAForce::ParticleInfo {
public:
    double radius;
    ParticleInfo() {
        radius = 0.0;
    }
    ParticleInfo(double radius) : radius(radius) {
    }
};

} // namespace OpenMM

#endif /*OPENMM_SASAFORCE_H_*/
