.. _other-features:

Other Features
##############


.. _periodic-boundary-conditions:

Periodic Boundary Conditions
****************************

Many Force objects support periodic boundary conditions.  They act as if space
were tiled with infinitely repeating copies of the system, then compute the
forces acting on a single copy based on the infinite periodic copies.  In most
(but not all) cases, they apply a cutoff so that each particle only interacts
with a single copy of each other particle.

OpenMM supports triclinic periodic boxes.  This means the periodicity is defined
by three vectors, :math:`\mathbf{a}`\ , :math:`\mathbf{b}`\ , and
:math:`\mathbf{c}`\ .  Given a particle position, the infinite periodic copies
of that particle are generated by adding vectors of the form
:math:`i \mathbf{a}+j \mathbf{b}+k \mathbf{c}`\ , where :math:`i`\ ,
:math:`j`\ , and :math:`k` are arbitrary integers.

The periodic box vectors must be chosen to satisfy certain requirements.
Roughly speaking, :math:`\mathbf{a}`\ , :math:`\mathbf{b}`\ , and
:math:`\mathbf{c}` need to "mostly" correspond to the x, y, and z axes.  They
must have the form

.. math::
   \mathbf{a} = (a_x, 0, 0)

   \mathbf{b} = (b_x, b_y, 0)

   \mathbf{c} = (c_x, c_y, c_z)

It is always possible to put the box vectors into this form by rotating the
system until :math:`\mathbf{a}` is parallel to x and :math:`\mathbf{b}` lies in
the xy plane.

Furthermore, they must obey the following constraints:

.. math::
   a_x > 0, b_y > 0, c_z > 0

   a_x \ge 2 |b_x|

   a_x \ge 2 |c_x|

   b_y \ge 2 |c_y|

This effectively requires the box vectors to be specified in a particular
reduced form.  By forming combinations of box vectors (a process known as
"lattice reduction"), it is always possible to put them in this form without
changing the periodic system they represent.

These requirements have an important consequence: the periodic unit cell can
always be treated as an axis-aligned rectangular box of size
:math:`(a_x, b_y, c_z)`\ .  The remaining non-zero elements of the box vectors
cause the repeating copies of the system to be staggered relative to each other,
but they do not affect the shape or size of each copy.  The volume of the unit
cell is simply given by :math:`a_x b_y c_z`\ .

LocalEnergyMinimizer
********************

This provides an implementation of the L-BFGS optimization algorithm.
:cite:`Liu1989`  Given a Context specifying initial particle positions, it
searches for a nearby set of positions that represent a local minimum of the
potential energy.  Distance constraints are enforced during minimization by
adding a harmonic restraining force to the potential function.  The strength of
the restraining force is steadily increased until the minimum energy
configuration satisfies all constraints to within the tolerance specified by the
Context's Integrator.

XMLSerializer
*************

This provides the ability to “serialize” a System, Force, Integrator, or State
object to a portable XML format, then reconstruct it again later.  When
serializing a System, the XML data contains a complete copy of the entire system
definition, including all Forces that have been added to it.

Here are some examples of uses for this class:

#. A model building utility could generate a System in memory, then serialize it
   to a file on disk.  Other programs that perform simulation or analysis could
   then reconstruct the model by simply loading the XML file.
#. When running simulations on a cluster, all model construction could be done
   on a single node.  The Systems and Integrators could then be encoded as XML,
   allowing them to be easily transmitted to other nodes.


XMLSerializer is a templatized class that, in principle, can be used to
serialize any type of object.  At present, however, only System, Force,
Integrator, and State are supported.

Force Groups
************

It is possible to split the Force objects in a System into groups.  Those groups
can then be evaluated independently of each other.  This is done by calling
:code:`setForceGroup()` on the Force.  Some Force classes also
provide finer grained control over grouping.  For example, NonbondedForce allows
direct space computations to be in one group and reciprocal space computations
in a different group.

The most important use of force groups is for implementing multiple time step
algorithms with CustomIntegrator.  For example, you might evaluate the slowly
changing nonbonded interactions less frequently than the quickly changing bonded
ones.  This can be done by putting the slow and fast forces into separate
groups, then using a :class:`MTSIntegrator` or :class:`MTSLangevinIntegrator`
that evaluates the groups at different frequencies.

Another important use is to define forces that are not used when integrating
the equations of motion, but can still be queried efficiently.  To do this,
call :code:`setIntegrationForceGroups()` on the :class:`Integrator`.  Any groups
omitted will be ignored during simulation, but can be queried at any time by
calling :code:`getState()`.

.. _virtual-sites:

Virtual Sites
*************

A virtual site is a particle whose position is computed directly from the
positions of other particles, not by integrating the equations of motion.  An
important example is the “extra sites” present in 4 and 5 site water models.
These particles are massless, and therefore cannot be integrated.  Instead,
their positions are computed from the positions of the massive particles in the
water molecule.

Virtual sites are specified by creating a VirtualSite object, then telling the
System to use it for a particular particle.  The VirtualSite defines the rules
for computing its position.  It is an abstract class with subclasses for
specific types of rules.  They are:

* TwoParticleAverageSite: The virtual site location is computed as a weighted
  average of the positions of two particles:

.. math::
   \mathbf{r}={w}_{1}\mathbf{r}_{1}+{w}_{2}\mathbf{r}_{2}

* ThreeParticleAverageSite: The virtual site location is computed as a weighted
  average of the positions of three particles:

.. math::
   \mathbf{r}={w}_{1}\mathbf{r}_{1}+{w}_{2}\mathbf{r}_{2}+{w}_{3}\mathbf{r}_{3}

* OutOfPlaneSite: The virtual site location is computed as a weighted average
  of the positions of three particles and the cross product of their relative
  displacements:

.. math::
   \mathbf{r}=\mathbf{r}_{1}+{w}_{12}\mathbf{r}_{12}+{w}_{13}\mathbf{r}_{13}+{w}_\mathit{cross}\left(\mathbf{r}_{12}\times \mathbf{r}_{13}\right)
..

  where :math:`\mathbf{r}_{12} = \mathbf{r}_{2}-\mathbf{r}_{1}` and
  :math:`\mathbf{r}_{13} = \mathbf{r}_{3}-\mathbf{r}_{1}`\ .  This allows
  the virtual site to be located outside the plane of the three particles.

* LocalCoordinatesSite: The locations of several other particles are used to compute a local
  coordinate system, and the virtual site is placed at a fixed location in that coordinate
  system.  The number of particles used to define the coordinate system is user defined.
  The origin of the coordinate system and the directions of its x and y axes
  are each specified as a weighted sum of the locations of the other particles:

.. math::
   \mathbf{o}={w}^{o}_{1}\mathbf{r}_{1} + {w}^{o}_{2}\mathbf{r}_{2} + ...

   \mathbf{dx}={w}^{x}_{1}\mathbf{r}_{1} + {w}^{x}_{2}\mathbf{r}_{2} + ...

   \mathbf{dy}={w}^{y}_{1}\mathbf{r}_{1} + {w}^{y}_{2}\mathbf{r}_{2} + ...

   \mathbf{dz}=\mathbf{dx}\times \mathbf{dy}
..

   These vectors are then used to construct a set of orthonormal coordinate axes as follows:

.. math::
   \mathbf{\hat{x}}=\mathbf{dx}/|\mathbf{dx}|

   \mathbf{\hat{z}}=\mathbf{dz}/|\mathbf{dz}|

   \mathbf{\hat{y}}=\mathbf{\hat{z}}\times \mathbf{\hat{x}}
..

   Finally, the position of the virtual site is set to

.. math::
   \mathbf{r}=\mathbf{o}+p_1\mathbf{\hat{x}}+p_2\mathbf{\hat{y}}+p_3\mathbf{\hat{z}}
..

* SymmetrySite: The virtual site location is computed by applying a rotation and
  translation to the position of a single other particle:

.. math::
   \mathbf{r}=\mathbf{R} \mathbf{r}_0 + \mathbf{v}
..

  where :math:`\mathbf{r}_0` is the position of the original particle,
  :math:`\mathbf{R}` is a rotation matrix, and :math:`\mathbf{v}` is a fixed
  vector.  The calculation can be performed either in Cartesian coordinates (for
  example, to build a symmetric copy of a molecule to create a dimer) or in
  fractional coordinates as defined by the periodic box (for example, to create
  a crystallographic unit cell consisting of multiple identical molecules
  related by symmetry).

Random Numbers with Stochastic Integrators and Forces
*****************************************************

OpenMM includes many stochastic integrators and forces that make extensive use
of random numbers. It is impossible to generate truly random numbers on a
computer like you would with a dice roll or coin flip in real life---instead
programs rely on pseudo-random number generators (PRNGs) that take some sort of
initial "seed" value and steps through a sequence of seemingly random numbers.

The exact implementation of the PRNGs is not important (in fact, each platform
may have its own PRNG whose performance is optimized for that hardware).  What
*is* important, however, is that the PRNG must generate a uniform distribution
of random numbers between 0 and 1. Random numbers drawn from this distribution
can be manipulated to yield random integers in a desired range or even a random
number from a different type of probability distribution function (e.g., a
normal distribution).

What this means is that the random numbers used by integrators and forces within
OpenMM cannot have any discernible pattern to them.  Patterns can be induced in
PRNGs in two principal ways:

1. The PRNG uses a bad algorithm with a short period.
2. Two PRNGs are started using the same seed

All PRNG algorithms in common use are periodic---that is their sequence of
random numbers repeats after a given *period*, defined by the number of "unique"
random numbers in the repeating pattern.  As long as this period is longer than
the total number of random numbers your application requires (preferably by
orders of magnitude), the first problem described above is avoided. All PRNGs
employed by OpenMM have periods far longer than any current simulation can cycle
through.

Point two is far more common in biomolecular simulation, and can result in very
strange artifacts that may be difficult to detect. For example, with Langevin
dynamics, two simulations that use the same sequence of random numbers appear to
synchronize in their global movements.\ :cite:`Uberuaga2004`\
:cite:`Sindhikara2009` It is therefore very important that the stochastic forces
and integrators in OpenMM generate unique sequences of pseudo-random numbers not
only within a single simulation, but between two different simulations of the
same system as well (including any restarts of previous simulations).

Every stochastic force and integrator that does (or could) make use of random
numbers has two instance methods attached to it: :meth:`getRandomNumberSeed()`
and :meth:`setRandomNumberSeed(int seed)`. If you set a unique random seed for
two different simulations (or different forces/integrators if applicable),
OpenMM guarantees that the generated sequences of random numbers will be
different (by contrast, no guarantee is made that the same seed will result in
identical random number sequences).

Since breaking simulations up into pieces and/or running multiple replicates of
a system to obtain more complete statistics is common practice, a new strategy
has been employed for OpenMM versions 6.3 and later with the aim of trying to
ensure that each simulation will be started with a unique random seed. A random
seed value of 0 (the default) will cause a unique random seed to be generated
when a new :class:`Context` is instantiated.

Prior to the introduction of this feature, deserializing a serialized
:class:`System` XML file would result in each stochastic force or integrator
being assigned the same random seed as the original instance that was
serialized. If you use a :class:`System` XML file generated by a version of
OpenMM older than 6.3 to start a new simulation, you should manually set the
random number seed of each stochastic force or integrator to 0 (or another
unique value).
