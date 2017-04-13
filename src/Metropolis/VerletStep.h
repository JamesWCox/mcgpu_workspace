/**
 * VerletStep.h
 *
 * A subclass of SimulationStep that uses a "verlet list" for energy
 * calculations
 *
 */

#ifndef METROPOLIS_VERLETSTEP_H
#define METROPOLIS_VERLETSTEP_H

#include "SimulationStep.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

class VerletStep: public SimulationStep {
    private:

        // Used to avoid needing to copy numMolecules from the GPU and avoid unecessary data transfers 
        int NUM_MOLS;   
        int VERLET_SIZE;
        int VACOORDS_SIZE;
        thrust::host_vector<int> h_verletList;
        thrust::host_vector<Real> h_verletAtomCoords;
        thrust::device_vector<int> d_verletList;
        thrust::device_vector<Real> d_verletAtomCoords;

        void checkOutsideSkinLayer(int molIdx);
        void resizeThrustVectors();
        void freeMemory();
        void CreateVerletList();

    public:
        explicit VerletStep(SimBox* box): SimulationStep(box),
                                    h_verletList(0),
                                    h_verletAtomCoords(0),
                                    d_verletList(0),
                                    d_verletAtomCoords(0) {

            NUM_MOLS = box->numMolecules;
            VERLET_SIZE = pow(NUM_MOLS, 2);
            VACOORDS_SIZE = NUM_DIMENSIONS * NUM_MOLS;
        }

        virtual ~VerletStep();
        virtual Real calcSystemEnergy(Real &subLJ, Real &subCharge, int numMolecules);

       /**
        * Determines the energy contribution of a particular molecule.
        * @param currMol The index of the molecule to calculate the contribution
        * @param startMol The index of the molecule to begin searching from to 
        *                      determine interaction energies
        * @param verletList The host_vector<int> containing the indexes of molecules
        *                      in range for each molecule 
        * @return The total energy of the box (discounts initial lj / charge energy)
        */
        virtual Real calcMolecularEnergyContribution(int currMol, int startMol);
        virtual void changeMolecule(int molIdx, SimBox *box);
        virtual void rollback(int molIdx, SimBox *box);
};

/**
 * VerletCalcs namespace
 *
 * Contains logic for calculations used by the VerletStep class.
 * Although logically related to the VerletStep class, these need
 * to be seperated to accurately run on the GPU.
 */
namespace VerletCalcs {

    /**
     *
     */
    template <typename T>
    struct EnergyContribution {

        T currMol;
        SimBox* sb;
        EnergyContribution(const T& _currMol, SimBox* _sb) {
            currMol = _currMol;
            sb = _sb;
        }

        __host__ __device__
        Real operator()(const T neighbor) const;
    };

    /**
     *
     */
    template <typename T>
    struct UpdateVerletList {
        __host__ __device__
        bool operator()(const T i, const Real* vaCoords, SimBox* sb) const;
    };

    /**
     * Sets the verlet neighbors for molID's portion of verlet list memory
     * For each molecule
     */
    template <typename T>
    struct NewVerletList {

        SimBox* sb;
        NewVerletList(SimBox* _sb) {
            sb = _sb;
        }

        __host__ __device__
       T* operator()() ;
    };

    /**
     *
     */
    //__global__
    //void energyContribution_Kernel(int currMol, int startMol, SimBox* sb, int* verletList, int verletListLength);
     /**
      * Determines whether or not two molecule's primaryIndexes are
      * within the cutoff range of one another and calculates the 
      * energy between them (if within range)
      *
      * @param m1 Molecule 1 
      * @param m2 Molecule 2
      * @param sb The SimBox from which data is to be used 
      * @return The total energy between two molecules 
      */
    __host__ __device__
    Real calcMoleculeInteractionEnergy (int m1, int m2, SimBox* sb);

}

#endif
