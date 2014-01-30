/*****************************************************************************
 *                                                                           *
 *             Mixed-mode OpenMP/MPI MicroBenchmark Suite - Version 1.0      *
 *                                                                           *
 *                            produced by                                    *
 *                                                                           *
 *                Mark Bull, Jim Enright and Fiona Reid                      *
 *                                                                           *
 *                                at                                         *
 *                                                                           *
 *                Edinburgh Parallel Computing Centre                        *
 *                                                                           *
 *   email: markb@epcc.ed.ac.uk, fiona@epcc.ed.ac.uk                         *
 *                                                                           *
 *                                                                           *
 *              Copyright 2012, The University of Edinburgh                  *
 *                                                                           *
 *                                                                           *
 *  Licensed under the Apache License, Version 2.0 (the "License");          *
 *  you may not use this file except in compliance with the License.         *
 *  You may obtain a copy of the License at                                  *
 *                                                                           *
 *      http://www.apache.org/licenses/LICENSE-2.0                           *
 *                                                                           *
 *  Unless required by applicable law or agreed to in writing, software      *
 *  distributed under the License is distributed on an "AS IS" BASIS,        *
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *
 *  See the License for the specific language governing permissions and      *
 *  limitations under the License.                                           *
 *                                                                           *
 ****************************************************************************/

/*-----------------------------------------------------------*/
/* Contains variables and routines for the parallel          */
/* environment.                                              */
/* Routines to setup the MPI and OpenMP programming          */
/* environment.                                              */
/* Header file = parallelEnvironment.h                       */
/*-----------------------------------------------------------*/
#include "parallelEnvironment.h"
#include "output.h"

int myThreadID;
int numThreads;


/*-----------------------------------------------------------*/
/* initParallelEnv			                                 */
/*                                                           */
/* Initialises the MPI and OpenMP environments.              */
/* Finds the total number of MPI processes and               */
/* OpenMP threads.                                           */
/* Also finds the ID of each MPI process and OpenMP thread.  */
/*-----------------------------------------------------------*/

#ifndef THREADS
#define THREADS 1
#endif

int initParallelEnv(){
    omp_set_num_threads(THREADS);

    /* Setup MPI programming environment */
	MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &threadSupport);

	comm = MPI_COMM_WORLD;
	MPI_Comm_size(comm, &numMPIprocs);
	MPI_Comm_rank(comm, &myMPIRank);

	/*Find the number of bytes for an int */
	sizeInteger = sizeof(int);

	/* Find the processor name of each MPI process */
    MPI_Get_processor_name(myProcName, &procNameLen);

	/* Use processor name to create a communicator
	 * across node boundaries.
	 */
	setupCommunicators();

	/* setup OpenMP programming environment */
    #pragma omp parallel shared(numThreads,globalIDarray,myMPIRank)
   {
	   numThreads = omp_get_num_threads();
	   myThreadID = omp_get_thread_num();

	   /* Allocate space for globalIDarray */
        #pragma omp single
       {
           globalIDarray = (int *)malloc(numThreads * sizeof(int));
       }

	   /*calculate the globalID for each thread */
	   globalIDarray[myThreadID] = (myMPIRank * numThreads) + myThreadID;
   }
    MPI_Barrier(comm);

    gaspi_config_t config;
    GASPI(config_get(&config));
    config.qp_count = THREADS;
    GASPI(config_set(config));
    /* GASPI setup */
    GASPI(proc_init(GASPI_BLOCK));

    gaspi_rank_t totalRanks;
    GASPI(proc_num(&totalRanks));

    gaspi_rank_t rank;
    GASPI(proc_rank(&rank));

    gaspi_number_t q_num;
    GASPI(queue_num(&q_num));
    assert (q_num == THREADS);

    GASPI(barrier (GASPI_GROUP_ALL, GASPI_BLOCK));
    // ok, we will continue to use the MPI ranks, just make sure GASPI and MPI ranks are identical
    // this is not guaranteed, so depending on the setup this may fail.
    assert (totalRanks == numMPIprocs);
    assert (rank == myMPIRank);

   /* set parallel info in benchmark report type */
   setParallelInfo(numMPIprocs,threadSupport,numThreads);

return 0;
}

/*-----------------------------------------------------------*/
/* finaliseParallelEnv                                       */
/*                                                           */
/* Closes the MPI programming environment.                   */
/*                                                         	 */
/*-----------------------------------------------------------*/
int finaliseParallelEnv(){
    GASPI(proc_term(GASPI_BLOCK));
	/* finalise the MPI programming environment */
    MPI_Finalize();
    /*free the space created for globalIDarray...*/
    free(globalIDarray);

return 0;
}

/*-----------------------------------------------------------*/
/* findRank                                                  */
/*                                                           */
/* Finds the MPI ranks which will take part in the pingping  */
/* or pingpong benchmarks based on the numbers read from the */
/* input file.                                               */
/*-----------------------------------------------------------*/
int findRank(int rankIn){
	int CalcRank;

    /* Figure out actual MPI rank */
    if (rankIn < 0){
    	CalcRank = numMPIprocs + rankIn;
    }
    else{
    	CalcRank = rankIn;
    }

    /* Check if findRank is too big or still -ve */
    if (CalcRank > (numMPIprocs-1)){
       printf("Warning: Rank input greater than total process count.\n");
       printf("Using Rank = %d ", numMPIprocs-1);
       CalcRank = numMPIprocs - 1;
    }
    else if(CalcRank < 0){
    	printf("Warning: MPI process offset greater than total process count.\n");
    	printf("Using Rank = 0 ");
    	CalcRank = 0;
    }

    return CalcRank;
}

/*-----------------------------------------------------------*/
/* findNeighbourRanks                          				 */
/*															 */
/* This creates a cartesian topology and finds the left      */
/* and right neighbours of each process.                     */
/*-----------------------------------------------------------*/
int findNeighbours(){
	int dims[1];
	int periods[1];
	int reorder;

	/* find a good process distribution */
	dims[0] = 0; /* zero so that dims_create tries to rearrange */
	MPI_Dims_create(numMPIprocs, 1, dims);

	/* set periods equal to TURE for periodic boundary conditions ... */
	periods[0] = TRUE;
	/* ...and reorder = FALSE */
	reorder = FALSE;

	/* Create the cartesian topology */
	MPI_Cart_create(comm, 1, dims, periods, reorder, &commCart);

	/* Find the ranks of the left and right neighbour */
	MPI_Cart_shift(commCart, 0, 1, &leftNeighbour, &rightNeighbour);

    MPI_Isend(&myMPIRank, 1, MPI_INT, leftNeighbour, TAG, commCart, &requestArray[0]);

    MPI_Isend(&myMPIRank, 1, MPI_INT, rightNeighbour, TAG, commCart, &requestArray[1]);

    MPI_Irecv(&myGASPIrightNeighbour, 1, MPI_INT, leftNeighbour, TAG, commCart, &requestArray[2]);

    MPI_Irecv(&myGASPIleftNeighbour, 1, MPI_INT, rightNeighbour, TAG, commCart, &requestArray[3]);

    MPI_Waitall(4, requestArray, statusArray);

	return 0;
}

/*-----------------------------------------------------------*/
/* benchmarkSupport                                          */
/*                                                           */
/* This function compares the level of thread support        */
/* needed by a particular benchmark with the level provided  */
/* by the implementation.                                    */
/*-----------------------------------------------------------*/
int benchmarkSupport(int required){
    return TRUE;

    /*int benchSupport;

    if (required <= threadSupport){
       benchSupport = TRUE;
    }
    else {
       benchSupport = FALSE;
    }

    return benchSupport;*/
}

/*-----------------------------------------------------------*/
/* compareProcNames                                          */
/*                                                           */
/* Compares the names of 2 processes to check if they are on */
/* the same node or not.                                     */
/*-----------------------------------------------------------*/
int compareProcNames(int rankA, int rankB){
	int sameNode;
	char recvProcName[MPI_MAX_PROCESSOR_NAME];

	/* Rank B sends procName to Rank A */
	if (myMPIRank == rankB){
		MPI_Send(myProcName, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, rankA, TAG, comm);
	}
	else if (myMPIRank == rankA){
		MPI_Recv(recvProcName, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, rankB, TAG, comm, &status);
		/* Rank B compares the two processor names */
		if (strcmp(myProcName,recvProcName) == 0){
			sameNode = TRUE;
		}
		else{
			sameNode = FALSE;
		}
	}

	/* Rank A then broadcasts its sameNode value to the other processes */
	MPI_Bcast(&sameNode, 1, MPI_INT, rankA, comm);

	return sameNode;
}

/*-----------------------------------------------------------*/
/* setupCommunicators                                        */
/*                                                           */
/* This creates two new communicators.                       */
/* The first gives a local communicator for processes on     */
/* the same node.                                            */
/* The second uses the local rank to give a communicator     */
/* across node boundaries.                                   */
/*                                                           */
/* e.g. for 16 nodes each with 2 processors, this routine    */
/*      will give 16 local communicators of size 2 and       */
/*      2 communicators of size 2 across nodes.              */
/*-----------------------------------------------------------*/
int setupCommunicators(){
	int procHash;

	/* Get hash from processor name */
	procHash = procNameToHash();

	/* Comm_split using procHash as colour to get
	 * local communicator.
	 */
	MPI_Comm_split(comm, procHash, 0, &localComm);

	/* Find ranks of processes in localComm */
	MPI_Comm_rank(localComm, &localCommRank);

	/* Find the size of localComm (for use in calculating multi datasize) */
	MPI_Comm_size(localComm, &localCommSize);

	/* Use localRank as colour to get communicator across nodes. */
	MPI_Comm_split(comm, localCommRank, 0, &crossComm);

	/* Find ranks of processes in crossComm */
	MPI_Comm_rank(crossComm, &crossCommRank);

    MPI_Barrier(comm);

	return 0;
}

/*-----------------------------------------------------------*/
/* procNameToHash                                            */
/*                                                           */
/* Creates an integer hash for each process.                 */
/* Each process on the same node will have the same hash     */
/* value.                                                    */
/*-----------------------------------------------------------*/
int procNameToHash(){
	int procHash,i;

	/* Initialise hash to 0 */
	procHash = 0;

	for (i=0; i<procNameLen; i++){
		procHash = (7 * procHash) + (int)(myProcName[i]);
	}

	return procHash;
}


/*-----------------------------------------------------------*/
/* exchangeWorldRanks                                        */
/*                                                           */
/* Finds the MPI_COMM_WORLD ranks of the processes           */
/* participating in the multi-pingpong and multi-pingping    */
/* benchmarks.                                               */
/*-----------------------------------------------------------*/
int exchangeWorldRanks(int nodeA, int nodeB, int *otherWorldRank){
	int destRank;

	if (crossCommRank == nodeA){
		destRank = nodeB;
	}
	else if (crossCommRank == nodeB){
		destRank = nodeA;
	}

	if (crossCommRank == nodeA || crossCommRank == nodeB){
		/* Start send of comm_world rank to destRank in crossComm */
		MPI_Isend(&myMPIRank, 1, MPI_INT, destRank, TAG, crossComm, &requestID);

		/* Then wait for message from destRank and store in otherWorldRank. */
		MPI_Recv(otherWorldRank, 1, MPI_INT, destRank, TAG, crossComm, &status);

		MPI_Wait(&requestID, &status);
	}

	return 0;
}

/*-----------------------------------------------------------*/
/* sendProcName                                              */
/*                                                           */
/* Sends the processor name from processes in destNode       */
/* of crossComm to srcNode.                                  */
/*-----------------------------------------------------------*/
void sendProcName(int destNode, int srcNode, char *destProcName){

	/* MPI processes under srcNode of crossComm send their
	 * processor name to destNode.
	 */
	if (crossCommRank == srcNode){
		MPI_Send(myProcName, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, \
				destNode, TAG, crossComm);
	}
	else if (crossCommRank == destNode){
		MPI_Recv(destProcName, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, \
				srcNode, TAG, crossComm, &status);
	}
}

/*-----------------------------------------------------------*/
/* checkCrossCommBalance                                     */
/*                                                           */
/* Checks if there's a balance in the number of processes    */
/* in crossComm nodes.                                       */
/*-----------------------------------------------------------*/
int crossCommBalance(int nodeA, int nodeB){
	int localCommSize, otherLocalCommSize;
	int crossCommBalance;

	/* Find the size of localComm */
	MPI_Comm_size(localComm, &localCommSize);

	/* Master process on nodeB sends localCommSize */
	if ((crossCommRank == nodeB) && (localCommRank == 0)){
		MPI_Send(&localCommSize, 1, MPI_INT, nodeA, TAG, crossComm);
	}
	/* Master process on nodeA... */
	else if ((crossCommRank == nodeA) && (localCommRank == 0)){
		/* 1) receives nodeB's localCommSize */
		MPI_Recv(&otherLocalCommSize, 1, MPI_INT, nodeB, TAG, \
				crossComm, &status);

		/* 2) Test for balance by comparing otherLocalCommSize
		 * to localCommSize.
		 */
		if (localCommSize == otherLocalCommSize){
			/* Set balance to TRUE */
			crossCommBalance = TRUE;
		}
		else{
			crossCommBalance = FALSE;
		}

		/* 3) Send balance to master commWorld process.
		 * Only need explicit send if commWorld is not same
		 * process as master process on nodeA.
		 */
		if (myMPIRank != 0){
			MPI_Send(&crossCommBalance, 1, MPI_INT, 0, TAG, comm);
		}
	}

	/* Master commWorld process.. */
	if (myMPIRank == 0){
		/* Receives balance variable if not same process as
		 * master process on nodeA.
		 */
		if ((crossCommRank != nodeA) && (localCommRank != 0)){
			MPI_Recv(&crossCommRank, 1, MPI_INT, MPI_ANY_SOURCE, \
					TAG, comm, &status);
		}
	}

	/* Broadcast balance to all processes */
	MPI_Bcast(&crossCommBalance, 1, MPI_INT, 0, comm);

	return crossCommBalance;
}
