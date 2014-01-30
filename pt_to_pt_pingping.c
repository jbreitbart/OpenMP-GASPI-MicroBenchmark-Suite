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
/* Contains the point-to-point pingping mixed mode           */
/* OpenMP/MPI benchmarks.                                    */
/* This includes: -masteronly pingping                       */
/*                -funnelled pingping                        */
/*                -multiple pingping                         */
/*-----------------------------------------------------------*/
#include "pt_to_pt_pingping.h"


/*-----------------------------------------------------------*/
/* pingPing                                    				 */
/*                                                           */
/* Driver subroutine for the pingping benchmark.             */
/*-----------------------------------------------------------*/
int pingPing(int benchmarkType){
    MPI_Barrier(comm);
    int dataSizeIter;
	int sameNode;

	pingRankA = PPRanks[0];
	pingRankB = PPRanks[1];

	/* Check if pingRankA and pingRankB are on the same node */
	sameNode = compareProcNames(pingRankA, pingRankB);


	if (myMPIRank == 0){
		/* print message saying if benchmark is inter or intra node */
		printNodeReport(sameNode,pingRankA,pingRankB);
		/* then print report column headings. */
		printBenchHeader();
	}

	/* initialise repsToDo to defaultReps at start of benchmark */
	repsToDo = defaultReps;

	/* Loop over data sizes */
	dataSizeIter = minDataSize; /* initialise dataSizeIter to minDataSize */
	while (dataSizeIter <= maxDataSize){
		/* set sizeofBuffer */
		sizeofBuffer = dataSizeIter * numThreads;

		/* Allocate space for main data arrays */
		allocatePingpingData(sizeofBuffer);

		/* warm-up for benchmarkType */
		if (benchmarkType == MASTERONLY){
			/* Masteronly warmp sweep */
			masteronlyPingping(warmUpIters, dataSizeIter);
		}
		else if (benchmarkType == FUNNELLED){
			/* perform funnelled warm-up sweep */
			funnelledPingping(warmUpIters, dataSizeIter);
		}
		else if (benchmarkType == MULTIPLE){
			multiplePingping(warmUpIters, dataSizeIter);
		}

        GASPI(barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
        /* perform verification test for the pingping */
		testPingping(sizeofBuffer, dataSizeIter);
        MPI_Barrier(comm);

		/* Initialise benchmark */
		benchComplete = FALSE;

		/* keep executing benchmark until target time is reached */
		while (benchComplete != TRUE){
			/* Start the timer...MPI_Barrier to synchronise */
			GASPI(barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
			startTime = MPI_Wtime();

			if (benchmarkType == MASTERONLY){
				/* execute for repsToDo repetitions */
				masteronlyPingping(repsToDo, dataSizeIter);
			}
			else if (benchmarkType == FUNNELLED){
				funnelledPingping(repsToDo, dataSizeIter);
			}
			else if (benchmarkType == MULTIPLE){
				multiplePingping(repsToDo, dataSizeIter);
			}

			/* Stop the timer...MPI_Barrier to synchronise processes */
			GASPI(barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
			finishTime = MPI_Wtime();
			totalTime = finishTime - startTime;

			/* Call repTimeCheck function to test if target time is reached */
			if (myMPIRank==0){
			  benchComplete = repTimeCheck(totalTime, repsToDo);
			}
            MPI_Barrier(comm);

			/* Ensure all procs have the same value of benchComplete */
			/* and repsToDo */
			MPI_Bcast(&benchComplete, 1, MPI_INT, 0, comm);
			MPI_Bcast(&repsToDo, 1, MPI_INT, 0, comm);
		}

		/* Master process sets benchmark results */
		if (myMPIRank == 0){
			setReportParams(dataSizeIter, repsToDo, totalTime);
			printReport();
		}

		/* Free the allocated space for the main data arrays */
		freePingpingData();

		/* Update dataSize before the next iteration */
		dataSizeIter = dataSizeIter * 2; /* double data size */
	}

	return 0;
}

/*-----------------------------------------------------------*/
/* masteronlyPingping										 */
/* 															 */
/* Two processes send a message to each other using the      */
/* MPI_Isend, MPI_Recv and MPI_Wait routines.				 */
/* Inter-process communication takes place outside of the    */
/* parallel region.											 */
/*-----------------------------------------------------------*/
int masteronlyPingping(int totalReps, int dataSize){
	int repIter, i;
	int destRank;

    char* seg_ptr;
    GASPI(segment_ptr(0, (gaspi_pointer_t*)&seg_ptr));

    /* set destRank to ID of other process */
	if (myMPIRank == pingRankA){
		destRank = pingRankB;
	}
	else if (myMPIRank == pingRankB){
		destRank = pingRankA;
	}

    for (repIter = 0; repIter < totalReps; repIter++){

		if (myMPIRank == pingRankA || myMPIRank == pingRankB){

			/* Each thread writes its globalID to pingSendBuf
			 * using a PARALLEL DO directive.
			 */
            #pragma omp parallel for  \
                private(i) \
                shared(pingSendBuf,dataSize,sizeofBuffer,globalIDarray) \
                schedule(static,dataSize)
			for (i=0; i<sizeofBuffer; i++){
				pingSendBuf[i] = globalIDarray[myThreadID];
            }

			/* Process calls non-bloacking send to start transfer of pingSendBuf
			 * to other process.
			 */
#if ONE_SIDED
            GASPI(write_notify(0, (char*)pingSendBuf-seg_ptr, destRank,
                               0, (char*)pingRecvBuf-seg_ptr, sizeof(int)*sizeofBuffer,
                               0, 1, 0, GASPI_BLOCK));
            //MPI_Isend(pingSendBuf, sizeofBuffer, MPI_INT, destRank, \
            //		TAG, comm, &requestID);

			/* Process then waits for message from other process. */
            gaspi_notification_id_t first;
            gaspi_notification_t old;
            GASPI(notify_waitsome(0, 0, 1, &first, GASPI_BLOCK));
            GASPI(notify_reset(0, first, &old));
            //MPI_Recv(pingRecvBuf, sizeofBuffer, MPI_INT, destRank, \
            //		TAG, comm, &status);

			/* Finish the Send operation with an MPI_Wait */
            GASPI(wait(0, GASPI_BLOCK));
            //MPI_Wait(&requestID, &status);

            GASPI(notify(0, destRank, numThreads+myThreadID, 1, myThreadID, GASPI_BLOCK));
#else
            if (myMPIRank == pingRankA) {
                gaspi_rank_t source;
                GASPI(passive_send(0, (char*)pingSendBuf-seg_ptr, destRank, sizeof(int)*sizeofBuffer, GASPI_BLOCK));
                GASPI(passive_receive(0, (char*)pingRecvBuf-seg_ptr, &source, sizeof(int)*sizeofBuffer, GASPI_BLOCK));
            }
            if (myMPIRank == pingRankB){
                gaspi_rank_t source;
                GASPI(passive_receive(0, (char*)pingRecvBuf-seg_ptr, &source, sizeof(int)*sizeofBuffer, GASPI_BLOCK));
                GASPI(passive_send(0, (char*)pingSendBuf-seg_ptr, destRank, sizeof(int)*sizeofBuffer, GASPI_BLOCK));
            }
#endif

            /* Each thread under the MPI process now reads its part of the
			 * received buffer.
			 */
            #pragma omp parallel for  \
                private(i) \
                shared(finalRecvBuf,dataSize,sizeofBuffer,pingRecvBuf) \
                schedule(static,dataSize)
			for (i=0; i<sizeofBuffer; i++){
				finalRecvBuf[i] = pingRecvBuf[i];
            }
#if ONE_SIDED
            GASPI(notify_waitsome(0, numThreads+myThreadID, 1, &first, GASPI_BLOCK));
            GASPI(notify_reset(0, first, &old));
#endif
		}
    }
#if ONE_SIDED
    GASPI(wait(0, GASPI_BLOCK));
#endif
    return 0;
}

/*-----------------------------------------------------------*/
/* funnelledPingPing                            		     */
/* 															 */
/* Two processes send a message to each other using the      */
/* MPI_Isend, MPI_Recv and MPI_Wait routines.                */
/* Inter-process communication takes place inside the        */
/* OpenMP parallel region.                                   */
/*-----------------------------------------------------------*/
int funnelledPingping(int totalReps, int dataSize){
	int repIter, i;
	int destRank;

    /* set destRank to ID of other process */
    if (myMPIRank == pingRankA){
    	destRank = pingRankB;
    }
    else if (myMPIRank == pingRankB){
    	destRank = pingRankA;
    }

    char* seg_ptr;
    GASPI(segment_ptr(0, (gaspi_pointer_t*)&seg_ptr));

	/* Open the parallel region */
    #pragma omp parallel \
        private(i, repIter) \
        shared(dataSize,sizeofBuffer,pingSendBuf,globalIDarray) \
        shared(pingRecvBuf,finalRecvBuf,status,requestID) \
        shared(destRank,comm,myMPIRank,pingRankA,pingRankB,totalReps)
	for (repIter = 0; repIter < totalReps; repIter++){


		if (myMPIRank == pingRankA || myMPIRank == pingRankB){

			/* Each thread writes its globalID to its part of
			 * pingSendBuf.
			 */
            #pragma omp for schedule(static,dataSize)
			for (i=0; i<sizeofBuffer; i++){
				pingSendBuf[i] = globalIDarray[myThreadID];
			}
            /* Implicit barrier here takes care of necessary synchronisation */

            #pragma omp master
			{
#if ONE_SIDED
                /* Process calls non-bloacking send to start transfer of pingSendBuf
                 * to other process.
                 */
                GASPI(write_notify(0, (char*)pingSendBuf-seg_ptr, destRank,
                                   0, (char*)pingRecvBuf-seg_ptr, sizeof(int)*sizeofBuffer,
                                   0, 1, 0, GASPI_BLOCK));
                //MPI_Isend(pingSendBuf, sizeofBuffer, MPI_INT, destRank, \
                //		TAG, comm, &requestID);

                /* Process then waits for message from other process. */
                gaspi_notification_id_t first;
                gaspi_notification_t old;
                GASPI(notify_waitsome(0, 0, 1, &first, GASPI_BLOCK));
                GASPI(notify_reset(0, first, &old));
                //MPI_Recv(pingRecvBuf, sizeofBuffer, MPI_INT, destRank, \
                //		TAG, comm, &status);

                /* Finish the Send operation with an MPI_Wait */
                GASPI(wait(0, GASPI_BLOCK));
                //MPI_Wait(&requestID, &status);

                GASPI(notify(0, destRank, numThreads+myThreadID, 1, myThreadID, GASPI_BLOCK));
#else
                if (myMPIRank == pingRankA) {
                    gaspi_rank_t source;
                    GASPI(passive_send(0, (char*)pingSendBuf-seg_ptr, destRank, sizeof(int)*sizeofBuffer, GASPI_BLOCK));
                    GASPI(passive_receive(0, (char*)pingRecvBuf-seg_ptr, &source, sizeof(int)*sizeofBuffer, GASPI_BLOCK));
                }
                if (myMPIRank == pingRankB){
                    gaspi_rank_t source;
                    GASPI(passive_receive(0, (char*)pingRecvBuf-seg_ptr, &source, sizeof(int)*sizeofBuffer, GASPI_BLOCK));
                    GASPI(passive_send(0, (char*)pingSendBuf-seg_ptr, destRank, sizeof(int)*sizeofBuffer, GASPI_BLOCK));
                }
#endif
            }

            /* Barrier needed to ensure master thread has completed transfer */
            #pragma omp barrier

			/* Each thread reads its part of the received buffer */
            #pragma omp for schedule(static,dataSize)
			for (i=0; i<sizeofBuffer; i++){
				finalRecvBuf[i] = pingRecvBuf[i];
			}

#if ONE_SIDED
            #pragma omp master
            {
                gaspi_notification_id_t first;
                gaspi_notification_t old;
                GASPI(notify_waitsome(0, numThreads+myThreadID, 1, &first, GASPI_BLOCK));
                GASPI(notify_reset(0, first, &old));
            }
#endif
		}
	}

#if ONE_SIDED
    GASPI(wait(0, GASPI_BLOCK));
#endif

    return 0;
}

/*-----------------------------------------------------------*/
/* multiplePingping                            				 */
/* 															 */
/* With this algorithm multiple threads take place in the    */
/* communication and computation.                            */
/* Each thread sends its portion of the pingSendBuf to the   */
/* other process using MPI_Isend/ MPI_Recv/ MPI_Wait         */
/* routines.                                                 */
/*-----------------------------------------------------------*/
int multiplePingping(int totalReps, int dataSize){
	int repIter, i;
	int destRank;
	int lBound;

    char* seg_ptr;
    GASPI(segment_ptr(0, (gaspi_pointer_t*)&seg_ptr));

    /* set destRank to ID of other process */
    if (myMPIRank == pingRankA){
    	destRank = pingRankB;
    }
    else if (myMPIRank == pingRankB){
    	destRank = pingRankA;
    }

    /* Open parallel region */
    #pragma omp parallel  \
        private(i,lBound,requestID,status,repIter) \
        shared(pingSendBuf,pingRecvBuf,finalRecvBuf,sizeofBuffer) \
        shared(destRank,myMPIRank,pingRankA,pingRankB,totalReps) \
        shared(dataSize,globalIDarray,comm)
    {
    for (repIter = 0; repIter < totalReps; repIter++){

    	if (myMPIRank == pingRankA || myMPIRank == pingRankB){

    		/* Calculate the lower bound of each threads
    		 * portion of the data arrays.
    		 */
    		lBound = (myThreadID * dataSize);

    		/* Each thread writes to its part of pingSendBuf */
            #pragma omp for nowait schedule(static,dataSize)
    		for (i=0; i<sizeofBuffer; i++){
    			pingSendBuf[i] = globalIDarray[myThreadID];
    		}

#if ONE_SIDED
            /* Each thread starts send of dataSize items of
             * pingSendBuf to process with rank = destRank.
             */
            GASPI(write_notify(0, (char*)&pingSendBuf[lBound]-seg_ptr, destRank,
                               0, (char*)&pingRecvBuf[lBound]-seg_ptr, sizeof(int)*dataSize,
                               myThreadID, 1, myThreadID, GASPI_TEST));
            //MPI_Isend(&pingSendBuf[lBound], dataSize, MPI_INT, destRank, \
            //		myThreadID, comm, &requestID);

            /* Thread then waits for message from destRank with
             * tag equal to it thread id.
             */
            gaspi_notification_id_t first;
            gaspi_notification_t old;
            GASPI(notify_waitsome(0, myThreadID, 1, &first, GASPI_BLOCK));
            GASPI(notify_reset(0, first, &old));
            //MPI_Recv(&pingRecvBuf[lBound], dataSize, MPI_INT, destRank, \
            //		myThreadID, comm, &status);

            /* Thread completes send using MPI_Wait */
            GASPI(wait(myThreadID, GASPI_BLOCK));
            //MPI_Wait(&requestID, &status);

            GASPI(notify(0, destRank, numThreads+myThreadID, 1, myThreadID, GASPI_BLOCK));

            /* Each thread reads its part of received buffer. */
            #pragma omp for nowait schedule(static,dataSize)
            for (i=0; i<sizeofBuffer; i++){
                finalRecvBuf[i] = pingRecvBuf[i];
            }
#else
            if (myMPIRank == pingRankA) {
                gaspi_rank_t source;
                GASPI(passive_send(0, (char*)&pingSendBuf[lBound]-seg_ptr, destRank, sizeof(int)*dataSize, GASPI_BLOCK));
                GASPI(passive_receive(0, (char*)&pingRecvBuf[lBound]-seg_ptr, &source, sizeof(int)*dataSize, GASPI_BLOCK));
            }
            if (myMPIRank == pingRankB) {
                gaspi_rank_t source;
                GASPI(passive_receive(0, (char*)&pingRecvBuf[lBound]-seg_ptr, &source, sizeof(int)*dataSize, GASPI_BLOCK));
                GASPI(passive_send(0, (char*)&pingSendBuf[lBound]-seg_ptr, destRank, sizeof(int)*dataSize, GASPI_BLOCK));
            }

            const int start = (pingRecvBuf[lBound]%omp_get_num_threads()) * dataSize;

            /* Each thread reads its part of the received buffer */
            //#pragma omp for nowait schedule(static,dataSize)
            for (i=0; i<dataSize; i++) {
                finalRecvBuf[start+i] = pingRecvBuf[lBound+i];
            }
#endif


#if ONE_SIDED
            // notify + wait to make sure the next iteration
            // starts after the previous one
            GASPI(notify_waitsome(0, numThreads+myThreadID, 1, &first, GASPI_BLOCK));
            GASPI(notify_reset(0, first, &old));
#endif
    	}
#if ONE_SIDED
#else
        GASPI(barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
#endif
    }
#if ONE_SIDED
    GASPI(wait(myThreadID, GASPI_BLOCK));
#endif
    }

    return 0;
}

/*-----------------------------------------------------------*/
/* allocatePingpingData                                      */
/*															 */
/* Allocates space for the main data arrays.                 */
/* Size of each array is specified by subroutine argument.   */
/*-----------------------------------------------------------*/
int allocatePingpingData(int sizeofBuffer){

    GASPI(segment_create(0, sizeofBuffer*sizeof(int)*3, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_ALLOC_DEFAULT));
    char* ptr;
    GASPI(segment_ptr(0, (gaspi_pointer_t*)&ptr));

    pingSendBuf = (int *)ptr;
    ptr +=sizeofBuffer * sizeof(int);
    pingRecvBuf = (int *)ptr;
    ptr += sizeofBuffer * sizeof(int);
    finalRecvBuf = (int *)ptr;

	return 0;
}

/*-----------------------------------------------------------*/
/* freePingpingData                                          */
/*															 */
/* Deallocates the storage space for the main data arrays.   */
/*-----------------------------------------------------------*/
int freePingpingData(){

    GASPI(segment_delete(0));

	return 0;
}

/*-----------------------------------------------------------*/
/* testPingping												 */
/*															 */
/* Verifies that the PingPing benchmark worked correctly.    */
/*-----------------------------------------------------------*/
int testPingping(int sizeofBuffer,int dataSize){
	int otherPingRank, i, testFlag, reduceFlag;
	int *testBuf;

	/* initialise testFlag to true (test passed) */
	testFlag = TRUE;

	/* Testing only needs to be done by pingRankA & pingRankB */
	if (myMPIRank == pingRankA || myMPIRank == pingRankB){
		/* allocate space for testBuf */
		testBuf = (int *)malloc(sizeofBuffer * sizeof(int));

		/* set the ID of other pingRank */
		if (myMPIRank == pingRankA){
			otherPingRank = pingRankB;
		}
		else if (myMPIRank == pingRankB){
			otherPingRank = pingRankA;
		}

		/* construct testBuf array with correct values.
		 * These are the values that should be in finalRecvBuf.
		 */
#pragma omp parallel for  \
	private(i) \
	shared(otherPingRank,numThreads,testBuf,dataSize,sizeofBuffer) \
	schedule(static,dataSize)
		for (i=0; i<sizeofBuffer; i++){
			/* calculate globalID of thread expected in finalRecvBuf
			 * This is done by using otherPingRank
			 */
			testBuf[i] = (otherPingRank * numThreads) + myThreadID;
		}

		/* compare each element of testBuf and finalRecvBuf */
		for (i=0; i<sizeofBuffer; i++){
			if (testBuf[i] != finalRecvBuf[i]){
				testFlag = FALSE;
			}
		}

		/* free space for testBuf */
		free(testBuf);
	}


	MPI_Reduce(&testFlag, &reduceFlag, 1, MPI_INT, MPI_LAND, 0, comm);

	/* Master process sets the testOutcome using testFlag. */
	 if (myMPIRank == 0){
		 setTestOutcome(reduceFlag);
	 }

	 return 0;
}
