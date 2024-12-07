/* Portions copyright (c) 2018 Stanford University and Simbios.
 * Contributors: Peter Eastman
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

#include "ReferenceSASAForce.h"
#include <bitset>
#include <iostream>
#include <stack>
#include "alphamol/Atoms.h"
#include "alphamol/Vertex.h"
#include "alphamol/Tetrahedron.h"
#include "alphamol/Edge.h"
#include "alphamol/Face.h"
#include "alphamol/delcx.h"
#include "alphamol/alfcx.h"
#include "alphamol/volumes.h"

DELCX delcx;
ALFCX alfcx;
VOLUMES volumes;

using namespace OpenMM;
using namespace std;

ReferenceSASAForce::ReferenceSASAForce() {
}

ReferenceSASAForce::~ReferenceSASAForce() {
}

double ReferenceSASAForce::calculateIxn(vector<Vec3>& atomCoordinates, vector<Vec3>& forces, vector<double>& radius, double scale) const {
    int numParticles = atomCoordinates.size();
    vector<double> coeff(numParticles, 1.0);
    vector<double> coord(3*numParticles);
    for (int i = 0; i < numParticles; i++) {
        coord[3*i] = atomCoordinates[i][0];
        coord[3*i+1] = atomCoordinates[i][1];
        coord[3*i+2] = atomCoordinates[i][2];
    }

    // Use AlphaMol to compute the surface area and its derivatives.

    vector<Vertex> vertices;
    vector<Tetrahedron> tetra;
    delcx.setup(numParticles, coord.data(), radius.data(), coeff.data(), coeff.data(), coeff.data(), coeff.data(), vertices, tetra);
    delcx.regular3D(vertices, tetra);
    double alpha = 0;
    alfcx.alfcx(alpha, vertices, tetra);
    vector<Edge> edges;
    vector<Face> faces;
    alfcx.alphacxEdges(tetra, edges);
    alfcx.alphacxFaces(tetra, faces);
    double Surf, WSurf, Vol, WVol, Mean, WMean, Gauss, WGauss;
    int nfudge = 8;
    vector<double> ballwsurf(numParticles+nfudge);
    vector<double> dsurf(3*(numParticles+nfudge), 0.0);
    vector<double> ballwvol(numParticles+nfudge);
    vector<double> dvol(3*(numParticles+nfudge), 0.0);
    vector<double> ballwmean(numParticles+nfudge);
    vector<double> dmean(3*(numParticles+nfudge), 0.0);
    vector<double> ballwgauss(numParticles+nfudge);
    vector<double> dgauss(3*(numParticles+nfudge), 0.0);
    volumes.ball_dvolumes(vertices, tetra, edges, faces, &WSurf, &WVol, &WMean, &WGauss, &Surf, &Vol, &Mean, &Gauss,
            ballwsurf.data(), ballwvol.data(), ballwmean.data(), ballwgauss.data(), dsurf.data(), dvol.data(), dmean.data(), dgauss.data(), 1);

    // Copy the derivatives to the output array.

    for (int i = 0; i < numParticles; i++)
        forces[i] -= scale*Vec3(dsurf[3*i], dsurf[3*i+1], dsurf[3*i+2]);
    return Surf*scale;
}
