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
/* Contains the point-to-point halo exchange mixed mode      */
/* OpenMP/MPI benchmarks.                                    */
/* This includes: -masteronly haloexchange                   */
/*                -funnelled haloexchange                    */
/*                -multiple haloexchange                     */
/*-----------------------------------------------------------*/
#include "pt_to_pt_haloexchange.h"
#include <errno.h>
#include <string.h>

/*-----------------------------------------------------------*/
/* haloExchange                                		         */
/*															 */
/* Driver subroutine for the haloExchange benchmark.         */
/*-----------------------------------------------------------*/
int haloExchange(int benchmarkType){
	int dataSizeIter;

	/* find the ranks of the left and right neighbour */
	findNeighbours();

	/* initialise repsToDo to defaultReps */
	repsToDo = defaultReps;

	/* Start loop over data sizes */
	dataSizeIter = minDataSize; /* Initialise dataSizeIter */

    MPI_Barrier(comm);
	while (dataSizeIter <= maxDataSize){
		/* set sizeofBuffer */
		sizeofBuffer = dataSizeIter * numThreads;

		/*Allocate space for the main data arrays */
		allocateHaloexchangeData(sizeofBuffer);

		/* perform benchmark warm-up */
		if (benchmarkType == MASTERONLY){
			masteronlyHaloexchange(warmUpIters, dataSizeIter);
		}
		else if (benchmarkType == FUNNELLED){
			funnelledHaloexchange(warmUpIters, dataSizeIter);
		}
		else if (benchmarkType == MULTIPLE){
			multipleHaloexchange(warmUpIters, dataSizeIter);
		}

        GASPI(barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
        /* Each process performs a verification test */
		testHaloexchange(sizeofBuffer, dataSizeIter);
        MPI_Barrier(comm);

		/*Initialise the benchmark */
		benchComplete = FALSE;

		/*Execute benchmark until target time is reached */
		while (benchComplete != TRUE){
			/*Start timer */
			GASPI(barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
			startTime = MPI_Wtime();

			/*Execute benchmarkType for repsToDo repetitions*/
			if (benchmarkType == MASTERONLY){
				masteronlyHaloexchange(repsToDo, dataSizeIter);
			}
			else if (benchmarkType == FUNNELLED){
				funnelledHaloexchange(repsToDo, dataSizeIter);
			}
			else if (benchmarkType == MULTIPLE){
				multipleHaloexchange(repsToDo, dataSizeIter);
			}

			/*Stop timer */
			GASPI(barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
			finishTime = MPI_Wtime();
			totalTime = finishTime - startTime;

            MPI_Barrier(comm);
            /* Test if target time is reached with the number of reps */
			if (myMPIRank==0){
			  benchComplete = repTimeCheck(totalTime, repsToDo);
			}
			/* Ensure all procs have the same value of benchComplete */
			/* and repsToDo */
			MPI_Bcast(&benchComplete, 1, MPI_INT, 0, comm);
			MPI_Bcast(&repsToDo, 1, MPI_INT, 0, comm);
		}

		/* Master process sets benchmark results */
		if (myMPIRank == 0 ){
			setReportParams(dataSizeIter, repsToDo, totalTime);
			printReport();
		}

		/* Free allocated data */
		freeHaloexchangeData();

		/* Double dataSize and loop again */
		dataSizeIter = dataSizeIter * 2;
	}

	return 0;
}

/*-----------------------------------------------------------*/
/* masteronlyHaloexchange                      			     */
/*															 */
/* Each process exchanges a message with its left and        */
/* right neighbour.                                          */
/* Communication takes place outside of the parallel         */
/* region.                                                   */
/*-----------------------------------------------------------*/
int masteronlyHaloexchange(int totalReps, int dataSize){
    int repIter, i;

    char* seg_ptr;
    GASPI(segment_ptr(0, (gaspi_pointer_t*)&seg_ptr));

    for (repIter=0; repIter<totalReps; repIter++){

        /* Each thread writes its globalID to rightSendBuf
         * and leftSendBuf using a parallel for directive.
         */
        #pragma omp parallel for  \
          private(i) \
          shared(leftSendBuf,rightSendBuf,dataSize) \
          shared(sizeofBuffer,globalIDarray) \
          schedule(static,dataSize)
        for (i=0; i<sizeofBuffer; i++){
            leftSendBuf[i] = globalIDarray[myThreadID];
            rightSendBuf[i] = globalIDarray[myThreadID];
        }

#if ONE_SIDED
        /* Process starts send of data to leftNeighbour and
         * rightNeighbour using non-blocking send...
         */
        GASPI(write_notify(0, (char*)leftSendBuf-seg_ptr, myGASPIleftNeighbour,
                           0, (char*)leftRecvBuf-seg_ptr, sizeof(int)*sizeofBuffer,
                           0, 1, 0, GASPI_BLOCK));
        //MPI_Isend(leftSendBuf, sizeofBuffer, MPI_INT, leftNeighbour, \
        //          TAG, commCart, &requestArray[0]);

        GASPI(write_notify(0, (char*)rightSendBuf-seg_ptr, myGASPIrightNeighbour,
                           0, (char*)rightRecvBuf-seg_ptr, sizeof(int)*sizeofBuffer,
                           1, 1, 0, GASPI_BLOCK));
        //MPI_Isend(rightSendBuf, sizeofBuffer, MPI_INT, rightNeighbour, \
        //          TAG, commCart, &requestArray[1]);

        /* Process then waits for messages from leftNeighbour and rightNeighbour */
        //MPI_Irecv(leftRecvBuf, sizeofBuffer, MPI_INT, leftNeighbour, \
        //          TAG, commCart, &requestArray[2]);

        //MPI_Irecv(rightRecvBuf, sizeofBuffer, MPI_INT, rightNeighbour, \
        //          TAG, commCart, &requestArray[3]);

        gaspi_notification_id_t first;
        gaspi_notification_t old;
        GASPI(notify_waitsome(0, 0, 1, &first, GASPI_BLOCK));
        GASPI(notify_reset(0, first, &old));
        GASPI(notify_waitsome(0, 1, 1, &first, GASPI_BLOCK));
        GASPI(notify_reset(0, first, &old));

        #pragma omp parallel for  \
          private(i) \
          shared(leftRecvBuf,rightRecvBuf,dataSize,sizeofBuffer) \
          shared(finalLeftBuf,finalRightBuf) \
          schedule(static,dataSize)
        for (i=0; i<sizeofBuffer; i++){
            finalRightBuf[i] = rightRecvBuf[i];
            finalLeftBuf[i] = leftRecvBuf[i];
        }
#else
        int send = 0;
        int recv = 0;
        int wrong_order = 0;
        while (send != 2 || recv != 2) {
            if (send == 0) {
                int ret = gaspi_passive_send(0, (char*)leftSendBuf-seg_ptr, 0, sizeof(int)*sizeofBuffer, 10);
                if (ret == GASPI_SUCCESS) {
                    ++send;
                }
                if (ret == GASPI_ERROR) {
                    assert(0);
                }
            }
            if (send == 1) {
                int ret = gaspi_passive_send(0, (char*)rightSendBuf-seg_ptr, 0, sizeof(int)*sizeofBuffer, 10);
                if (ret == GASPI_SUCCESS) {
                    ++send;
                }
                if (ret == GASPI_ERROR) {
                    assert(0);
                }
            }
            if (recv == 0) {
                gaspi_rank_t source;
                int ret = gaspi_passive_receive(0, (char*)leftRecvBuf-seg_ptr, &source, sizeof(int)*sizeofBuffer, 10);
                if (ret == GASPI_SUCCESS) {
                    if (source != leftNeighbour) wrong_order = 1;
                    ++recv;
                }
                if (ret == GASPI_ERROR) {
                    printf("%s\n", strerror(errno));
                    assert(0);
                }
            }
            if (recv == 1) {
                gaspi_rank_t source;
                int ret = gaspi_passive_receive(0, (char*)rightRecvBuf-seg_ptr, &source, sizeof(int)*sizeofBuffer, 10);
                if (ret == GASPI_SUCCESS) {
                    if (source != rightNeighbour && wrong_order != 1) assert(0);
                    ++recv;
                }
                if (ret == GASPI_ERROR) {
                    printf("%s\n", strerror(errno));
                    assert(0);
                }
            }
        }

        if (wrong_order == 0) {
            #pragma omp parallel for  \
              private(i) \
              shared(leftRecvBuf,rightRecvBuf,dataSize,sizeofBuffer) \
              shared(finalLeftBuf,finalRightBuf) \
              schedule(static,dataSize)
            for (i=0; i<sizeofBuffer; i++){
                finalRightBuf[i] = rightRecvBuf[i];
                finalLeftBuf[i] = leftRecvBuf[i];
            }
        } else {
            #pragma omp parallel for  \
              private(i) \
              shared(leftRecvBuf,rightRecvBuf,dataSize,sizeofBuffer) \
              shared(finalLeftBuf,finalRightBuf) \
              schedule(static,dataSize)
            for (i=0; i<sizeofBuffer; i++){
                finalRightBuf[i] = leftRecvBuf[i];
                finalLeftBuf[i] = rightRecvBuf[i];
            }
        }
#endif


#if ONE_SIDED
        GASPI(notify(0, myGASPIrightNeighbour, 2, 1, 0, GASPI_BLOCK));
        GASPI(notify(0, myGASPIleftNeighbour, 3, 1, 0, GASPI_BLOCK));

        GASPI(notify_waitsome(0, 2, 1, &first, GASPI_BLOCK));
        GASPI(notify_reset(0, first, &old));
        GASPI(notify_waitsome(0, 3, 1, &first, GASPI_BLOCK));
        GASPI(notify_reset(0, first, &old));

        /* Finish the sends with an MPI_Waitall on the requests */
        //MPI_Waitall(4, requestArray, statusArray);
        GASPI(wait(0, GASPI_BLOCK));
#else
        GASPI(barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
#endif
    }

    return 0;
}

/*-----------------------------------------------------------*/
/* funnelledHaloexchange				     */
/*							     */
/* Each process exchanges a message with its left and        */
/* right neighbour.                                          */
/* Communication takes place by one thread inside of the     */
/* parallel region.                                          */
/*-----------------------------------------------------------*/
int funnelledHaloexchange(int totalReps, int dataSize){
    int repIter, i;

    char* seg_ptr;
    GASPI(segment_ptr(0, (gaspi_pointer_t*)&seg_ptr));

    /* Open the parallel region */
    #pragma omp parallel  \
      private(i,repIter) \
      shared(dataSize,sizeofBuffer,leftSendBuf,rightSendBuf) \
      shared(rightRecvBuf,leftRecvBuf,finalLeftBuf,finalRightBuf) \
      shared(globalIDarray,commCart,totalReps,requestArray,statusArray) \
      shared(leftNeighbour,rightNeighbour)
    {
        for (repIter=0; repIter<totalReps; repIter++){
            /* Each thread writes its globalID to rightSendBuf
             * and leftSendBuf.
             */
            #pragma omp for schedule(static,dataSize)
            for (i=0; i<sizeofBuffer; i++){
                leftSendBuf[i] = globalIDarray[myThreadID];
                rightSendBuf[i] = globalIDarray[myThreadID];
            }
            /* Implicit barrier here takes care of necessary synchronisation */

            #pragma omp master
            {
                /* Master thread starts send of data to left and right neighbours
                 * with a non-blocking send.
                 */
                GASPI(write_notify(0, (char*)leftSendBuf-seg_ptr, myGASPIleftNeighbour,
                                   0, (char*)leftRecvBuf-seg_ptr, sizeof(int)*sizeofBuffer,
                                   0, 1, 0, GASPI_BLOCK));
                //MPI_Isend(leftSendBuf, sizeofBuffer, MPI_INT, leftNeighbour, \
                //          TAG, commCart, &requestArray[0]);

                GASPI(write_notify(0, (char*)rightSendBuf-seg_ptr, myGASPIrightNeighbour,
                                   0, (char*)rightRecvBuf-seg_ptr, sizeof(int)*sizeofBuffer,
                                   1, 1, 0, GASPI_BLOCK));
                //MPI_Isend(rightSendBuf, sizeofBuffer, MPI_INT, rightNeighbour, \
                //          TAG, commCart, &requestArray[1]);

                gaspi_notification_id_t first;
                gaspi_notification_t old;
                GASPI(notify_waitsome(0, 0, 1, &first, GASPI_BLOCK));
                GASPI(notify_reset(0, first, &old));
                GASPI(notify_waitsome(0, 1, 1, &first, GASPI_BLOCK));
                GASPI(notify_reset(0, first, &old));

                /* Thread then starts receive of messages from leftNeighbour
                 * and rightNeighbour.
                 */
                //MPI_Irecv(leftRecvBuf, sizeofBuffer, MPI_INT, leftNeighbour, \
                //          TAG, commCart, &requestArray[2]);

                //MPI_Irecv(rightRecvBuf, sizeofBuffer, MPI_INT, rightNeighbour, \
                //          TAG, commCart, &requestArray[3]);
            }

            /*Barrier to ensure master thread has completed transfer. */
            #pragma omp barrier

            /* Each thread now reads its part of the left and right received buffers. */
            #pragma omp for schedule(static,dataSize)
            for(i=0; i<sizeofBuffer; i++){
                finalLeftBuf[i] = leftRecvBuf[i];
                finalRightBuf[i] = rightRecvBuf[i];
            }

            #pragma omp master
            {
                gaspi_notification_id_t first;
                gaspi_notification_t old;
                GASPI(notify(0, myGASPIrightNeighbour, 2, 1, 0, GASPI_BLOCK));
                GASPI(notify(0, myGASPIleftNeighbour, 3, 1, 0, GASPI_BLOCK));

                GASPI(notify_waitsome(0, 2, 1, &first, GASPI_BLOCK));
                GASPI(notify_reset(0, first, &old));
                GASPI(notify_waitsome(0, 3, 1, &first, GASPI_BLOCK));
                GASPI(notify_reset(0, first, &old));

                /* Finish the sends and receives with an MPI_Waitall on the requests */
                GASPI(wait(0, GASPI_BLOCK));
                //MPI_Waitall(4, requestArray, statusArray);
            }
            #pragma omp barrier
        }
    }
    return 0;
}

/*-----------------------------------------------------------*/
/* multipleHaloexchange                                      */
/*															 */
/* Each process exchanges a message with its left and        */
/* right neighbour.                                          */
/* All threads take part in the inter-porcess                */
/* communication.                                            */
/*-----------------------------------------------------------*/
int multipleHaloexchange(int totalReps, int dataSize){
    int repIter, i;
    int lBound;

    char* seg_ptr;
    GASPI(segment_ptr(0, (gaspi_pointer_t*)&seg_ptr));

    /* Open the parallel region */
    #pragma omp parallel  \
      private(i,requestArray,statusArray,lBound,repIter) \
      shared(dataSize,sizeofBuffer,leftSendBuf,rightSendBuf) \
      shared(rightRecvBuf,leftRecvBuf,finalLeftBuf,finalRightBuf) \
      shared(leftNeighbour,rightNeighbour,globalIDarray,commCart,totalReps)
    {
        for (repIter=0; repIter<totalReps; repIter++){
            /* Calculate lower bound for each thread */
            lBound = (myThreadID * dataSize);

            /* Each thread writes its globalID to rightSendBuf
             * and leftSendBuf.
             */
            #pragma omp for nowait schedule(static,dataSize)
            for (i=0; i<sizeofBuffer; i++){
                leftSendBuf[i] = globalIDarray[myThreadID];
                rightSendBuf[i] = globalIDarray[myThreadID];
            }

            /* Each thread starts send of dataSize items to leftNeighbour
             * and to rightNeighbour.
             */
            GASPI(write_notify(0, (char*)&leftSendBuf[lBound]-seg_ptr, myGASPIleftNeighbour,
                               0, (char*)&leftRecvBuf[lBound]-seg_ptr, sizeof(int)*dataSize,
                               myThreadID, 1, myThreadID, GASPI_BLOCK));
            //MPI_Isend(&leftSendBuf[lBound], dataSize, MPI_INT, leftNeighbour, \
            //          myThreadID, commCart, &requestArray[0]);

            GASPI(write_notify(0, (char*)&rightSendBuf[lBound]-seg_ptr, myGASPIrightNeighbour,
                               0, (char*)&rightRecvBuf[lBound]-seg_ptr, sizeof(int)*dataSize,
                               myThreadID+numThreads, 1, myThreadID, GASPI_BLOCK));
            //MPI_Isend(&rightSendBuf[lBound], dataSize, MPI_INT, rightNeighbour, \
            //          myThreadID, commCart, &requestArray[1]);


            /* Each Thread then starts receive of messages from leftNeighbour
             * and rightNeighbour.
             */
            //MPI_Irecv(&leftRecvBuf[lBound], dataSize, MPI_INT, leftNeighbour, \
            //          myThreadID, commCart, &requestArray[2]);

            //MPI_Irecv(&rightRecvBuf[lBound], dataSize, MPI_INT, rightNeighbour, \
            //          myThreadID, commCart, &requestArray[3]);

            gaspi_notification_id_t first;
            gaspi_notification_t old;
            GASPI(notify_waitsome(0, myThreadID, 1, &first, GASPI_BLOCK));
            GASPI(notify_reset(0, first, &old));
            GASPI(notify_waitsome(0, myThreadID+numThreads, 1, &first, GASPI_BLOCK));
            GASPI(notify_reset(0, first, &old));

            /* Finish the sends with an MPI_Waitall on the requests */
            //MPI_Waitall(4, requestArray, statusArray);

            /* Each thread now reads its part of the left and
             * right received buffers.
             */
            #pragma omp for nowait schedule(static,dataSize)
            for (i=0; i<sizeofBuffer; i++){
                finalLeftBuf[i] = leftRecvBuf[i];
                finalRightBuf[i] = rightRecvBuf[i];
            }

            GASPI(notify(0, myGASPIrightNeighbour, myThreadID+2*numThreads, 1, myThreadID, GASPI_BLOCK));
            GASPI(notify(0, myGASPIleftNeighbour, myThreadID+3*numThreads, 1, myThreadID, GASPI_BLOCK));

            GASPI(notify_waitsome(0, myThreadID+2*numThreads, 1, &first, GASPI_BLOCK));
            GASPI(notify_reset(0, first, &old));
            GASPI(notify_waitsome(0, myThreadID+3*numThreads, 1, &first, GASPI_BLOCK));
            GASPI(notify_reset(0, first, &old));

            /* Finish the sends and receives with an MPI_Waitall on the requests */
            GASPI(wait(myThreadID, GASPI_BLOCK));

        }
    }

    return 0;
}





/*-----------------------------------------------------------*/
/* allocateHaloexchangeData                   				 */
/*															 */
/* Allocate memory for the main data arrays in the           */
/* haloexchange.                                             */
/*-----------------------------------------------------------*/
int allocateHaloexchangeData(int sizeofBuffer){

    GASPI(segment_create(0, sizeofBuffer*sizeof(int)*6, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_ALLOC_DEFAULT));
    char* ptr;
    GASPI(segment_ptr(0, (gaspi_pointer_t*)&ptr));

    leftSendBuf = (int *)ptr;
    ptr += sizeofBuffer * sizeof(int);
    leftRecvBuf = (int *)ptr;
    ptr += sizeofBuffer * sizeof(int);
    rightSendBuf = (int *)ptr;
    ptr += sizeofBuffer * sizeof(int);
    rightRecvBuf = (int *)ptr;
    ptr += sizeofBuffer * sizeof(int);
    finalLeftBuf = (int *)ptr;
    ptr += sizeofBuffer * sizeof(int);
    finalRightBuf = (int *)ptr;

	return 0;
}

/*-----------------------------------------------------------*/
/* freeHaloexchangeData                                      */
/*															 */
/* Deallocates the storage space for the main data arrays.   */
/*-----------------------------------------------------------*/
int freeHaloexchangeData(){
    GASPI(segment_delete(0));
	return 0;
}

/*-----------------------------------------------------------*/
/* testHaloexchange                                          */
/*															 */
/* Verifies that the halo exchange benchmark worked          */
/* correctly.                                                */
/*-----------------------------------------------------------*/
int testHaloexchange(int sizeofBuffer, int dataSize){
	int i;
	int testFlag, reduceFlag;
	int *testLeftBuf, *testRightBuf;

	/* set testFlag to true */
	testFlag = TRUE;

	/*allocate space for testLeftBuf and testRightBuf */
	testLeftBuf = (int *)malloc(sizeofBuffer * sizeof(int));
	testRightBuf = (int *)malloc(sizeofBuffer * sizeof(int));

	/*construct testLeftBuf and testRightBuf with correct values */
#pragma omp parallel for  \
	private(i) \
	shared(leftNeighbour,rightNeighbour,numThreads) \
	shared(dataSize,sizeofBuffer,testLeftBuf,testRightBuf) \
	schedule(static,dataSize)
	for (i=0; i<sizeofBuffer; i++){
		/* Calculate globalID of thread expected in finalLeftBuf.. */
		testLeftBuf[i] = (leftNeighbour * numThreads) + myThreadID;
		/* ..and in finalRightBuf. */
		testRightBuf[i] = (rightNeighbour * numThreads) + myThreadID;
	}

	/* Compare.. */
	for (i=0; i<sizeofBuffer; i++){
		/* 1) values from left neighbour */
		if (testLeftBuf[i] != finalLeftBuf[i]){
			testFlag = FALSE;
		}
		/* 2) values from right neighbour */
		if (testRightBuf[i] != finalRightBuf[i]){
			testFlag = FALSE;
		}
	}

	MPI_Reduce(&testFlag, &reduceFlag, 1, MPI_INT, MPI_LAND, 0, comm);

	/* Master then sets testOutcome flag */
	if (myMPIRank == 0){
		setTestOutcome(reduceFlag);
	}

	/* free space for testLeftBuf and testRightBuf */
	free(testLeftBuf);
	free(testRightBuf);

	return 0;
}
