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
/* Contains the point-to-point multi-pingpong mixed mode     */
/* OpenMP/MPI benchmarks.                                    */
/* This includes: -masteronly multiPingpong                  */
/*                -funnelled multiPingpong                   */
/*                -multiple multiPingpong                    */
/*-----------------------------------------------------------*/
#include "pt_to_pt_multiPingpong.h"

/*-----------------------------------------------------------*/
/* multiPingPong                                             */
/*                                                           */
/* Driver subroutine for the multi-pingpong benchmark.       */
/*-----------------------------------------------------------*/
int multiPingPong(int benchmarkType){
    int dataSizeIter;
	int pongWorldRank;
	char pongProcName[MPI_MAX_PROCESSOR_NAME];
	int balance;

	pingNode = 0;
	pongNode = 1;

	/* Check if there's a balance in num of MPI processes
	  on pingNode and pongNode. */
    balance = crossCommBalance(pingNode, pongNode);
	/* If not balanced.. */
	if (balance == FALSE){
		/* ..master prints error */
		if (myMPIRank == 0){
			printBalanceError();
		}
		/* ..and all process exit function. */
		return 1;
    }

	/* Exchange MPI_COMM_WORLD ranks for processes in same crossComm */
	exchangeWorldRanks(pingNode, pongNode, &pongWorldRank);
    myGASPICrossCommPartner = pongWorldRank;

    /* Processes on pongNode send processor name to pingNode procs. */
    sendProcName(pingNode, pongNode, pongProcName);

	/* Print comm world ranks & processor name of processes
	 * taking part in multi-pingpong benchmark.
	 */
    printMultiProcInfo(pingNode, pongWorldRank, pongProcName);

	/* Barrier to ensure that all procs have completed
	 * printMultiProcInfo before prinring column headings.
	 */
    MPI_Barrier(comm);

	/* Master process then prints report column headings */
	if (myMPIRank == 0){
		printBenchHeader();
	}

    /* Initialise repsToDo to defaultReps at start of benchmark */
	repsToDo = defaultReps;
	dataSizeIter = minDataSize; /* initialise dataSizeIter to minDataSize */

	/* Loop over data sizes */
	while (dataSizeIter <= maxDataSize){
		/* set sizeofBuffer */
		sizeofBuffer = dataSizeIter * numThreads;

        /* Allocate space for the main data arrays */
		allocateMultiPingpongData(sizeofBuffer);

		/* warm-up */
		if (benchmarkType == MASTERONLY){
			/* Masteronly warm-up */
			masteronlyMultiPingpong(warmUpIters, dataSizeIter);
		}
		else if (benchmarkType == FUNNELLED){
			/* Funnelled warm-up sweep */
			funnelledMultiPingpong(warmUpIters, dataSizeIter);
		}
		else if (benchmarkType == MULTIPLE){
			/* Multiple pingpong warm-up */
			multipleMultiPingpong(warmUpIters, dataSizeIter);
		}
        GASPI(barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

        /* Verification test for multi-pingpong */
		testMultiPingpong(sizeofBuffer, dataSizeIter);
        MPI_Barrier(comm);

		/* Initialise benchmark */
		benchComplete = FALSE;

		/* Keep executing benchmark until target time is reached */
		while (benchComplete != TRUE){

			/* MPI_Barrier to synchronise processes.
			   Then start the timer. */
			GASPI(barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
			startTime = MPI_Wtime();

			if (benchmarkType == MASTERONLY){
				/* Execute masteronly multipingpong repsToDo times */
				masteronlyMultiPingpong(repsToDo, dataSizeIter);
			}
			else if (benchmarkType == FUNNELLED){
				/* Execute funnelled multipingpong */
				funnelledMultiPingpong(repsToDo, dataSizeIter);
			}
			else if (benchmarkType == MULTIPLE){
				multipleMultiPingpong(repsToDo, dataSizeIter);
			}

			/* Stop the timer..MPI_Barrier to synchronise processes
			 * for more accurate timing.
			 */
			GASPI(barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
            finishTime = MPI_Wtime();
			totalTime = finishTime - startTime;

            /* Call repTimeCheck to check if target time is reached. */
			if (myMPIRank==0){
			  benchComplete = repTimeCheck(totalTime, repsToDo);
			}
			/* Ensure all procs have the same value of benchComplete */
			/* and repsToDo */
			MPI_Bcast(&benchComplete, 1, MPI_INT, 0, comm);
			MPI_Bcast(&repsToDo, 1, MPI_INT, 0, comm);
            MPI_Barrier(comm);
        } /* End of loop to check if benchComplete is true */

		/* Master process sets benchmark results */
		if (myMPIRank == 0){
			setReportParams(dataSizeIter, repsToDo, totalTime);
			printReport();
		}

		/* Free the allocated space for the main data arrays */
		freeMultiPingpongData();

		/* Update dataSize before next iteration */
		dataSizeIter = dataSizeIter * 2;

	} /* end loop over data sizes */

	return 0;
}

/*-----------------------------------------------------------*/
/* masteronlyMultiPingpong                                   */
/*                                                           */
/* All MPI processes in crossComm = pingNode sends a single  */
/* fixed length message to the neighbouring process in       */
/* crossComm = pongNode.                                     */
/* The neighbouring processes then sends the message back    */
/* to the first process.                                     */
/*-----------------------------------------------------------*/
int masteronlyMultiPingpong(int totalReps, int dataSize){
	int repIter, i;

    char* seg_ptr;
    GASPI(segment_ptr(0, (gaspi_pointer_t*)&seg_ptr));
#if ONE_SIDED
    gaspi_number_t queue_size;
    GASPI(queue_size_max(&queue_size));
    gaspi_number_t queue_cur_size = 0;
#endif

    for (repIter = 1; repIter <= totalReps; repIter++){

		/* Threads under each MPI process with
		 * crossCommRank = pingNode write to pingSendBuf
		 * array with a PARALLEL FOR directive.
		 */
		if (crossCommRank == pingNode){

            #pragma omp parallel for  \
                private(i) \
                shared(pingSendBuf,dataSize,sizeofBuffer,globalIDarray) \
                schedule(static,dataSize)
			for (i=0; i<sizeofBuffer; i++){
				pingSendBuf[i] = globalIDarray[myThreadID];
			}

#if ONE_SIDED
            /* Each process with crossCommRank = pingNode sends
			 * buffer to MPI process with rank = pongNode in crossComm.
			 */
            GASPI(write_notify(0, (char*)pingSendBuf-seg_ptr, myGASPICrossCommPartner,
                               0, (char*)pingRecvBuf-seg_ptr, sizeof(int)*sizeofBuffer,
                               0, 1, 0, GASPI_BLOCK));
            queue_cur_size+=2;
            //MPI_Send(pingSendBuf, sizeofBuffer, MPI_INT, pongNode, TAG, crossComm);

			/* The processes then wait for a message from pong process
			 * and each thread reads its part of the received buffer.
			 */
            gaspi_notification_id_t first;
            gaspi_notification_t old;
            GASPI(notify_waitsome(0, 0, 1, &first, GASPI_BLOCK));
            GASPI(notify_reset(0, first, &old));
            //MPI_Recv(pongRecvBuf, sizeofBuffer, MPI_INT, pongNode, \
            //		TAG, crossComm, &status);
#else
            GASPI(passive_send(0, (char*)pingSendBuf-seg_ptr, myGASPICrossCommPartner, sizeof(int)*sizeofBuffer, GASPI_BLOCK));
            gaspi_rank_t source;
            GASPI(passive_receive(0, (char*)pongRecvBuf-seg_ptr, &source, sizeof(int)*sizeofBuffer, GASPI_BLOCK));
#endif
            #pragma omp parallel for  \
                private(i) \
                shared(pongRecvBuf,finalRecvBuf,dataSize,sizeofBuffer) \
                schedule(static,dataSize)
			for (i=0; i<sizeofBuffer; i++){
				finalRecvBuf[i] = pongRecvBuf[i];
			}
		}
		else if (crossCommRank == pongNode){

#if ONE_SIDED
            /* Each process with crossCommRank = pongNode receives
			 * the message from the pingNode processes.
			 */
            gaspi_notification_id_t first;
            gaspi_notification_t old;
            GASPI(notify_waitsome(0, 0, 1, &first, GASPI_BLOCK));
            GASPI(notify_reset(0, first, &old));
            //MPI_Recv(pingRecvBuf, sizeofBuffer, MPI_INT, pingNode,\
            //		TAG, crossComm, &status);
#else
            gaspi_rank_t source;
            GASPI(passive_receive(0, (char*)pingRecvBuf-seg_ptr, &source, sizeof(int)*sizeofBuffer, GASPI_BLOCK));
#endif
			/* Each thread copies its part of the received buffer
			 * to pongSendBuf.
			 */
            #pragma omp parallel for  \
                private(i) \
                shared(pongSendBuf,pingRecvBuf,dataSize,sizeofBuffer) \
                schedule(static,dataSize)
			for (i=0; i<sizeofBuffer; i++){
				pongSendBuf[i] = pingRecvBuf[i];
			}

#if ONE_SIDED
            /* The processes now send pongSendBuf to processes
			 * with crossCommRank = pingNode.
			 */
            GASPI(write_notify(0, (char*)pongSendBuf-seg_ptr, myGASPICrossCommPartner,
                               0, (char*)pongRecvBuf-seg_ptr, sizeofBuffer*sizeof(int),
                               0, 1, 0, GASPI_BLOCK
                              ));
            queue_cur_size+=2;
            //MPI_Send(pongSendBuf, sizeofBuffer, MPI_INT, pingNode, \
            //		TAG, crossComm);
#else
            GASPI(passive_send(0, (char*)pongSendBuf-seg_ptr, myGASPICrossCommPartner, sizeof(int)*sizeofBuffer, GASPI_BLOCK));
#endif
		}
#if ONE_SIDED
        if (queue_cur_size >= queue_size-2) {
            GASPI(wait(0, GASPI_BLOCK));
            queue_cur_size = 0;
        }
#endif
    } /* End repetitions loop */
#if ONE_SIDED
    GASPI(wait(0, GASPI_BLOCK));
#endif

	return 0;
}

/*-----------------------------------------------------------*/
/* funnelledMultiPingpong                                    */
/*                                                           */
/* All MPI processes in crossComm = pingNode sends a single  */
/* fixed length message to the neighbouring process in       */
/* crossComm = pongNode.                                     */
/* The neighbouring processes then sends the message back    */
/* to the first process.                                     */
/* All communication takes place within the OpenMP parallel  */
/* region for this benchmark.                                */
/*-----------------------------------------------------------*/
int funnelledMultiPingpong(int totalReps, int dataSize){
	int repIter, i;

    char* seg_ptr;
    GASPI(segment_ptr(0, (gaspi_pointer_t*)&seg_ptr));
#if ONE_SIDED
    gaspi_number_t queue_size;
    GASPI(queue_size_max(&queue_size));
    gaspi_number_t queue_cur_size = 0;
#endif
    /* Open the parallel region for threads */
    #pragma omp parallel  \
        private(i,repIter) \
        shared(pingNode,pongNode,pingSendBuf,pingRecvBuf) \
        shared(pongSendBuf,pongRecvBuf,finalRecvBuf,sizeofBuffer) \
        shared(dataSize,globalIDarray,crossComm,status) \
        shared(totalReps,myMPIRank,crossCommRank)
	{

		/* loop totalRep times */
		for (repIter = 1; repIter <= totalReps; repIter++){

			/* All threads under each MPI process with
			 * crossCommRank = pingNode write to pingSendBuf
			 * array using a parallel for directive.
			 */
			if (crossCommRank == pingNode){

                #pragma omp for schedule(static,dataSize)
				for (i=0; i<sizeofBuffer; i++){
					pingSendBuf[i] = globalIDarray[myThreadID];
				}
                /* Implicit barrier at end of omp for takes care of synchronisation */

				/* Master thread under each pingNode process sends
				 * buffer to corresponding MPI process in pongNode
				 * using crossComm.
				 */
                #pragma omp master
                {
#if ONE_SIDED
                    /* Each process with crossCommRank = pingNode sends
                     * buffer to MPI process with rank = pongNode in crossComm.
                     */
                    GASPI(write_notify(0, (char*)pingSendBuf-seg_ptr, myGASPICrossCommPartner,
                                       0, (char*)pingRecvBuf-seg_ptr, sizeof(int)*sizeofBuffer,
                                       0, 1, 0, GASPI_BLOCK
                                      ));
                    queue_cur_size+=2;
                    //MPI_Send(pingSendBuf, sizeofBuffer, MPI_INT, pongNode, TAG, crossComm);
                    /* The processes then wait for a message from pong process
                     * and each thread reads its part of the received buffer.
                     */
                    gaspi_notification_id_t first;
                    gaspi_notification_t old;
                    GASPI(notify_waitsome(0, 0, 1, &first, GASPI_BLOCK));
                    GASPI(notify_reset(0, first, &old));
                    //MPI_Recv(pongRecvBuf, sizeofBuffer, MPI_INT, pongNode, \
                    //		TAG, crossComm, &status);
#else
                    GASPI(passive_send(0, (char*)pingSendBuf-seg_ptr, myGASPICrossCommPartner, sizeof(int)*sizeofBuffer, GASPI_BLOCK));
                    gaspi_rank_t source;
                    GASPI(passive_receive(0, (char*)pongRecvBuf-seg_ptr, &source, sizeof(int)*sizeofBuffer, GASPI_BLOCK));
#endif
                }
                /* Barrier needed to wait for master thread to complete MPI_Recv */
                #pragma omp barrier

                /* Each thread then reads its part of the received buffer. */
                #pragma omp for schedule(static,dataSize)
				for (i=0; i<sizeofBuffer; i++){
					finalRecvBuf[i] = pongRecvBuf[i];
				}

			}
			else if (crossCommRank == pongNode){

				/* Master thread under each pongNode process receives
				 * the message from the pingNode processes.
				 */
                #pragma omp master
                {
#if ONE_SIDED
                    gaspi_notification_id_t first;
                    gaspi_notification_t old;
                    GASPI(notify_waitsome(0, 0, 1, &first, GASPI_BLOCK));
                    GASPI(notify_reset(0, first, &old));
                    //MPI_Recv(pingRecvBuf, sizeofBuffer, MPI_INT, pingNode,\
                    //		TAG, crossComm, &status);
#else
                    gaspi_rank_t source;
                    GASPI(passive_receive(0, (char*)pingRecvBuf-seg_ptr, &source, sizeof(int)*sizeofBuffer, GASPI_BLOCK));
#endif
                }
                /* Barrier needed to wait on master thread */
                #pragma omp barrier

                /* Each thread reads its part of the received buffer. */
                #pragma omp for schedule(static,dataSize)
				for (i=0; i<sizeofBuffer; i++){
					pongSendBuf[i] = pingRecvBuf[i];
				}
                /* Implicit barrier at end of omp for */

                /* Master threads send their pongSendBuf to processes
                 * with crossCommRank = pingNode.
                 */
                #pragma omp master
                {
#if ONE_SIDED
                    GASPI(write_notify(0, (char*)pongSendBuf-seg_ptr, myGASPICrossCommPartner, 0,
                                       (char*)pongRecvBuf-seg_ptr, sizeofBuffer*sizeof(int), 0, 1, 0, GASPI_BLOCK
                                      ));
                    queue_cur_size+=2;
                    //MPI_Send(pongSendBuf, sizeofBuffer, MPI_INT, pingNode, TAG, crossComm);
#else
                    GASPI(passive_send(0, (char*)pongSendBuf-seg_ptr, myGASPICrossCommPartner, sizeof(int)*sizeofBuffer, GASPI_BLOCK));
#endif
                }

			}
#if ONE_SIDED
            #pragma omp master
            if (queue_cur_size >= queue_size-2) {
                GASPI(wait(0, GASPI_BLOCK));
                queue_cur_size = 0;
            }
#endif
        } /* End of repetitions loop. */
	} /* End of parallel region */
#if ONE_SIDED
    GASPI(wait(0, GASPI_BLOCK));
#endif

	return 0;
}

/*-----------------------------------------------------------*/
/* multipleMultiPingpong                                     */
/*                                                           */
/* Multiple threads take place in the communication and      */
/* computation.                                              */
/* Each thread of all MPI processes in crossComm = pingNode  */
/* sends a portion of the message to the neighbouring        */
/* process in crossComm = pongNode.                          */
/* Each thread of the neighbouring processes then sends      */
/* the message back to the first process.                    */
/*-----------------------------------------------------------*/
int multipleMultiPingpong(int totalReps, int dataSize){
	int repIter, i;
	int lBound;

    char* seg_ptr;
    GASPI(segment_ptr(0, (gaspi_pointer_t*)&seg_ptr));
#if ONE_SIDED
    gaspi_number_t queue_size;
    GASPI(queue_size_max(&queue_size));
#endif
    /* Open parallel region for threads */
    #pragma omp parallel  \
        private(i,repIter,status,lBound) \
        shared(pingNode,pongNode,pingSendBuf,pingRecvBuf) \
        shared(pongSendBuf,pongRecvBuf,finalRecvBuf,sizeofBuffer) \
        shared(dataSize,globalIDarray,crossComm) \
        shared(totalReps,myMPIRank,crossCommRank)
	{
        gaspi_number_t queue_cur_size = 0;
        for (repIter=1; repIter<=totalReps; repIter++){ /* loop totalRep times */

			if (crossCommRank == pingNode){

				/* Calculate lower bound of data array for the thread */
				lBound = (myThreadID * dataSize);

				/* All threads write to its part of the pingBuf
				 * array using a parallel for directive.
				 */
                #pragma omp for nowait schedule(static,dataSize)
				for (i=0; i<sizeofBuffer; i++){
					pingSendBuf[i] = globalIDarray[myThreadID];
				}
                /* Implicit barrier at end of for not needed for multiple */

#if ONE_SIDED
                /* Each thread under ping process sends dataSize items
				 * to pongNode process in crossComm.
				 * myThreadID is used as tag to ensure data goes to
				 * correct place in buffer.
				 */
                GASPI(write_notify(0, (char*)&pingSendBuf[lBound]-seg_ptr, myGASPICrossCommPartner,
                                   0, (char*)&pingRecvBuf[lBound]-seg_ptr, sizeof(int)*dataSize,
                                   myThreadID, 1, myThreadID, GASPI_BLOCK
                                  ));
                queue_cur_size+=2;
                //MPI_Send(&pingSendBuf[lBound], dataSize, MPI_INT, pongNode, \
                //		myThreadID, crossComm);

				/* Thread then waits for a message from pongNode. */
                gaspi_notification_id_t first;
                gaspi_notification_t old;
                GASPI(notify_waitsome(0, myThreadID, 1, &first, GASPI_BLOCK));
                GASPI(notify_reset(0, first, &old));
                //MPI_Recv(&pongRecvBuf[lBound], dataSize, MPI_INT, pongNode, \
                //		myThreadID, crossComm, &status);

                /* Each thread reads its part of the received buffer. */
                #pragma omp for nowait schedule(static,dataSize)
                for (i=0; i<sizeofBuffer; i++){
                    finalRecvBuf[i] = pongRecvBuf[i];
                }
#else
                GASPI(passive_send(0, (char*)&pingSendBuf[lBound]-seg_ptr, myGASPICrossCommPartner, sizeof(int)*dataSize, GASPI_BLOCK));
                gaspi_rank_t source;
                GASPI(passive_receive(0, (char*)&pongRecvBuf[lBound]-seg_ptr, &source, sizeof(int)*dataSize, GASPI_BLOCK));

                const int start = (pongRecvBuf[lBound]%omp_get_num_threads()) * dataSize;
                const int end = start + dataSize;

                /* Each thread reads its part of the received buffer */
                //#pragma omp for nowait schedule(static,dataSize)
                for (i=start; i<end; i++) {
                    finalRecvBuf[i] = pongRecvBuf[i];
                }
#endif

			}
			else if (crossCommRank == pongNode){
				/* Calculate lower and upper bound of data array */
				lBound = (myThreadID * dataSize);

#if ONE_SIDED
				/* Each thread under pongRank receives a message from
				 * the ping process.
				 */
                gaspi_notification_id_t first;
                gaspi_notification_t old;
                GASPI(notify_waitsome(0, myThreadID, 1, &first, GASPI_BLOCK));
                GASPI(notify_reset(0, first, &old));
                //MPI_Recv(&pingRecvBuf[lBound], dataSize, MPI_INT, pingNode, \
                //		myThreadID, crossComm, &status);
#else
                gaspi_rank_t source;
                GASPI(passive_receive(0, (char*)&pingRecvBuf[lBound]-seg_ptr, &source, sizeof(int)*dataSize, GASPI_BLOCK));
#endif
				/* Each thread now copies its part of the received buffer
				 * to pongSendBuf.
				 */
                #pragma omp for nowait schedule(static,dataSize)
				for (i=0; i<sizeofBuffer; i++){
					pongSendBuf[i] = pingRecvBuf[i];
				}

#if ONE_SIDED
                /* Each thread now sends pongSendBuf to ping process. */
                GASPI(write_notify(0, (char*)&pongSendBuf[lBound]-seg_ptr, myGASPICrossCommPartner,
                                   0, (char*)&pongRecvBuf[lBound]-seg_ptr, sizeofBuffer*sizeof(int),
                                   0, 1, myThreadID, GASPI_BLOCK
                                  ));
                queue_cur_size+=2;
                //MPI_Send(&pongSendBuf[lBound], dataSize, MPI_INT, pingNode, \
                //		myThreadID, crossComm);
#else
                GASPI(passive_send(0, (char*)&pongSendBuf[lBound]-seg_ptr, myGASPICrossCommPartner, sizeof(int)*dataSize, GASPI_BLOCK));
#endif

			}
#if ONE_SIDED
            if (queue_cur_size >= queue_size-2) {
                GASPI(wait(myThreadID, GASPI_BLOCK));
                queue_cur_size = 0;
            }
#endif
        } /* End repetitions loop */
#if ONE_SIDED
        GASPI(wait(myThreadID, GASPI_BLOCK));
#endif
	} /* End parallel region */

	return 0;
}

/*-----------------------------------------------------------*/
/* allocateMultiPingpongData                                 */
/*                                                           */
/* Allocates space for the main data arrays.                 */
/* Size of each array is specified by subroutine argument.   */
/*-----------------------------------------------------------*/
int allocateMultiPingpongData(int sizeofBuffer){

    GASPI(segment_create(0, sizeofBuffer*sizeof(int)*5, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_ALLOC_DEFAULT));
    char* ptr;
    GASPI(segment_ptr(0, (gaspi_pointer_t*)&ptr));

    pingSendBuf = (int *)ptr;
    ptr += sizeof(int) * sizeofBuffer;
    pongRecvBuf = (int *)ptr;
    ptr += sizeof(int) * sizeofBuffer;
    finalRecvBuf = (int *)ptr;
    ptr += sizeof(int) * sizeofBuffer;
    pingRecvBuf = (int *)ptr;
    ptr += sizeof(int) * sizeofBuffer;
    pongSendBuf = (int *)ptr;

    return 0;
}

/*-----------------------------------------------------------*/
/* freeMultiPingpongData                                     */
/*                                                           */
/* Deallocates the storage space for the main data arrays.   */
/*-----------------------------------------------------------*/
int freeMultiPingpongData(){

    GASPI(segment_delete(0));

	return 0;
}

/*-----------------------------------------------------------*/
/* testMultiPingpong                                         */
/*                                                           */
/* Verifies the the multi pingpong benchmark worked          */
/* correctly.                                                */
/*-----------------------------------------------------------*/
int testMultiPingpong(int sizeofBuffer, int dataSize){
	int i;
	int testFlag, localTestFlag;

	/* Initialise localTestFlag to true */
	localTestFlag = TRUE;

	/* All processes with crossCommRank = pingNode check
	 * if multi-pingpong worked ok.
	 */
	if (crossCommRank == pingNode){

		/* allocate space for testBuf */
		testBuf = (int *)malloc(sizeof(int) * sizeofBuffer);

		/* Construct testBuf array with correct values.
		 * These are the values that should be in finalRecvBuf.
		 */
#pragma omp parallel for  \
	private(i) \
	shared(testBuf,dataSize,sizeofBuffer,globalIDarray)\
	schedule(static,dataSize)

		for (i=0; i<sizeofBuffer; i++){
			testBuf[i] = globalIDarray[myThreadID];
		}


		/* Compare each element of testBuf and finalRecvBuf */
		for (i=0; i<sizeofBuffer; i++){
			if (testBuf[i] != finalRecvBuf[i]){
				localTestFlag = FALSE;
			}
		}

		/* Free space for testBuf */
		free(testBuf);
	}

	/* Reduce localTestFlag to master */
	MPI_Reduce(&localTestFlag, &testFlag, 1, MPI_INT,MPI_LAND, 0, comm);

	/* Master then sets testOutcome using reduceFlag */
	if (myMPIRank == 0){
		setTestOutcome(testFlag);
	}

	return 0;
}

