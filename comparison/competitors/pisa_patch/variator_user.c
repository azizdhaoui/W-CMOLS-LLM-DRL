/*========================================================================
  PISA  (www.tik.ee.ethz.ch/pisa/)
 
  ========================================================================
  Computer Engineering (TIK)
  ETH Zurich
 
  ========================================================================
  PISALIB 
  
  Functions the user must implement
  
  C file.
  
  file: variator_user.h
  author: Marco Laumanns, laumanns@tik.ee.ethz.ch

  revision by: Stefan Bleuler, bleuler@tik.ee.ethz.ch
  last change: $date$
   
  ========================================================================
*/

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "variator.h"
#include "variator_user.h"

/*--------------------| global variable definitions |-------------------*/

/*==== declared in variator_user.h used in other files as well ====*/

char *log_file = "knapsack_error.log"; /**** Changed for knapsack.*/

char paramfile[FILE_NAME_LENGTH]; /* file with local parameters */


/*==== only used in this file ====*/

double *weights;
double *profits;
double *profitSums;
double *capacities;
int *selectOrder;


/* local parameters from paramfile*/
int seed;   /* seed for random number generator */
int length; /* length of the binary string */
int maxgen; /* maximum number of generations (stop criterion) */
int gen;
char outfile[FILE_NAME_LENGTH]; /* output file for last population */

int mutation_type; /* 0 = no mutation
		      1 = one bit mutation
		      2 = independent bit mutation */

int recombination_type; /* 0 = no recombination
			   1 = one point crossover
			   2 = uniform crossover */

/* probability that individual is mutated */
double mutat_prob; 

 /* probability that 2 individual are recombined */
double recom_prob;

/* probability, that bit is turned when mutation occurs only used for
 * independent bit mutation */
double bit_turn_prob;



/*-------------------------| individual |-------------------------------*/

void free_individual(individual *ind) 
/* Frees the memory for one indiviual.

   post: memory for ind is freed
*/
{
     /**********| added for KNAPSACK |**************/
     
     if (ind == NULL)
          return;
     
     free(ind->bit_string);
     free(ind->objective_value);
     
     /**********| addition for KNAPSACK end |*******/
     
     free(ind);
}

double get_objective_value(int identity, int i)
/* Gets objective value of an individual.

   pre: 0 <= i <= dimension - 1 (dimension is the number of objectives)

   post: Return value == the objective value number 'i' in individual '*ind'.
         If no individual with ID 'identity' return value == -1. 
*/   
{
     /**********| added for KNAPSACK |**************/
     individual *temp_ind;
     /**********| addition for KNAPSACK end |*******/
     
     double objective_value = -1.0;

     assert(0 <= i && i < dimension); /* asserting the pre-condition */
     
     /**********| added for KNAPSACK |**************/
    
     if (i < 0 || i > (dimension - 1))
	  return(-1);
     
     temp_ind = get_individual(identity);
     if (temp_ind == NULL)
	 return(-1);
     
     objective_value = temp_ind->objective_value[i];
     
     /**********| addition for KNAPSACK end |*******/
  
     return (objective_value);
}

/*-------------------------| statemachine functions |-------------------*/

int state0() 
/* Do what needs to be done in state 0.

   pre: The global variable 'paramfile' contains the name of the
        parameter file specified on the commandline.
        The global variable 'alpha' contains the number of indiviuals
        you need to generate for the initial population.
                
   post: Optionally read parameter specific for the module.
         Optionally do some initialization.
         Initial population created.
         Information about initial population written to the ini file
         using write_ini().
         Return value == 0 if successful,
                      == 1 if unspecified errors happened,
                      == 2 if file reading failed.
*/
{
     /**********| added for KNAPSACK |**************/
     int i;
     /**********| addition for KNAPSACK end |*******/

     
     int result; /* stores return values of called functions */
     int *initial_population; /* storing the IDs of the individuals */
     initial_population = (int *) malloc(alpha * sizeof(int)); 
     if (initial_population == NULL)
     {
          log_to_file(log_file, __FILE__, __LINE__, "variator out of memory");
          return (1);
     }

     /**********| added for KNAPSACK |**************/
     result = read_local_parameters();
     
     if (result != 0)
     { 
          log_to_file(log_file, __FILE__, __LINE__,
                      "couldn't read local parameters");
          return (1);
     }
     
     /* create first alpha individuals */
     for(i = 0; i < alpha; i++)
     {
          initial_population[i] = add_individual(new_individual());
          if(initial_population[i] == -1)
               return(1);
     } 

     gen = 1;
     
     /**********| addition for KNAPSACK end |*******/

     result = write_ini(initial_population);
     if (result != 0)
     { 
          log_to_file(log_file, __FILE__, __LINE__,
                      "couldn't write ini");
          free(initial_population);
          return (1);
     }

     free(initial_population);
     return (0);
}



int state2()
/* Do what needs to be done in state 2.

   pre: The global variable 'mu' contains the number of indiviuals
        you need to read using 'read_sel()'.
        The global variable 'lambda' contains the number of individuals
        you need to create by variation of the individuals specified the
        'sel' file.
        
   post: Optionally call read_arc() in order to delete old uncessary
         individuals from the global population.
         read_sel() called
         'lambda' children generated from the 'mu' parents
         Children added to the global population using add_individual().
         Information about children written to the 'var' file using
         write_var().
         Return value == 0 if successful,
                      == 1 if unspecified errors happened,
                      == 2 if file reading failed.
*/
{
     int *parent_identities, *offspring_identities; /* array of identities */
     int result; /* stores return values of called functions */

     parent_identities = (int *) malloc(mu * sizeof(int)); 
     if (parent_identities == NULL)
     {
          log_to_file(log_file, __FILE__, __LINE__, "variator out of memory");
          return (1);
     }

     offspring_identities = (int *) malloc(lambda * sizeof(int)); 
     if (offspring_identities == NULL)
     {
          log_to_file(log_file, __FILE__, __LINE__, "variator out of memory");
          return (1);
     }
     
     result = read_sel(parent_identities);
     if (result != 0) /* if some file reading error occurs, return 2 */
          return (2);

     result = read_arc(); 
     if (result != 0) /* if some file reading error occurs, return 2 */
          return (2);

     /**********| added for KNAPSACK |**************/

     result = variate(parent_identities, offspring_identities);
     if (result != 0)
          return (1);

     gen++;

     /**********| addition for KNAPSACK end |*******/


     result = write_var(offspring_identities);
     if (result != 0)
     { 
          log_to_file(log_file, __FILE__, __LINE__,
                      "couldn't write var");
          free(offspring_identities);
          free(parent_identities);
          return (1);
     }

     free(offspring_identities);
     free(parent_identities);
     return (0);
}
 

int state4() 
/* Do what needs to be done in state 4.

   pre: State 4 means the variator has to terminate.

   post: Free all memory.
         Return value == 0 if successful,
                      == 1 if unspecified errors happened,
                      == 2 if file reading failed.
*/
{
     /**********| added for KNAPSACK |**************/
     
     int result;
     result = read_arc();
   
     if (0 == result) /* arc file correctly read
                         this means it was not read before,
                         e.g., in a reset. */
     {
        write_output_file();
     }

     free(weights);
     weights = NULL;

     free(profits);
     profits = NULL;
     
     free(profitSums);
     profits = NULL;
     
     free(capacities);
     capacities = NULL;
     
     free(selectOrder);
     selectOrder = NULL;
 
     /**********| addition for KNAPSACK end |*******/
     
     return (0);
}


int state7()
/* Do what needs to be done in state 7.

   pre: State 7 means that the selector has just terminated.

   post: You probably don't need to do anything, just return 0.
         Return value == 0 if successful,
                      == 1 if unspecified errors happened,
                      == 2 if file reading failed.
*/
{
     return(0);  
}


int state8()
/* Do what needs to be done in state 8.

   pre: State 8 means that the variator needs to reset and get ready to
        start again in state 0.

   post: Get ready to start again in state 0. 
         Return value == 0 if successful,
                      == 1 if unspecified errors happened,
                      == 2 if file reading failed.
*/
{
     /**********| added for KNAPSACK |**************/

   int result;
   
   gen = 1;
     
   result = read_arc();

   if (0 == result) /* arc file correctly read
                       this means it was not read before */
   {
      write_output_file();
   }
   
   /*all of the following are allocated again in read_local_parameters() */
   
   free(weights);
   weights = NULL;
   
   free(profits);
   profits = NULL;
   
   free(profitSums);
   profitSums = NULL;
   
   free(capacities);
   capacities = NULL;
   
   free(selectOrder);
   selectOrder = NULL;
   
     /**********| addition for KNAPSACK end |*******/
     
     return (0);
}


int state11()
/* Do what needs to be done in state 11.

   pre: State 11 means that the selector has just reset and is ready
        to start again in state 1.

   post: You probably don't need to do anything, just return 0.
         Return value == 0 if successful,
                      == 1 if unspecified errors happened,
                      == 2 if file reading failed.
*/
{
     return (0);  
}


int is_finished()
/* Tests if ending criterion of your algorithm applies.

   post: return value == 1 if optimization should stop
         return value == 0 if optimization should continue
*/
{
     /**********| added for KNAPSACK |**************/
     return (gen >= maxgen);
     /**********| addition for KNAPSACK end |*******/
}


/**********| added for KNAPSACK |**************/

/* temporary array used for sorting items */
static double* profitWeightRatios;

/* function used for sorting items in decreasing order */
int cmpItems(const void*  itemPtr1, const void*  itemPtr2)
{
    if (profitWeightRatios[*((int*) itemPtr1)] >
	profitWeightRatios[*((int*) itemPtr2)])  return -1;
    if (profitWeightRatios[*((int*) itemPtr2)] > 
	profitWeightRatios[*((int*) itemPtr1)])  return 1;
    return 0;
}

int RandomInt(int  min, int  max)
/* generates a random integer */
{
  return (int) (min + (rand() % (max - min + 1)));
} /* RandomInt */

int read_local_parameters()
{
     FILE *fp;
     int result;
     char str[CFG_NAME_LENGTH];
     int i, j;

     /* reading parameter file with parameters for selection */
     fp = fopen(paramfile, "r"); 
     assert(fp != NULL);

     if(dimension < 0)
     {
          log_to_file(log_file, __FILE__, 
                      __LINE__, "can't handle that dimension");
          return(1);
     } 

     if(mu != lambda)
     {
          log_to_file(log_file, __FILE__, 
                      __LINE__, "can't handle mu != lambda");
          return(1);
     }

     fscanf(fp, "%s", str);
     assert(strcmp(str, "seed") == 0);
     fscanf(fp, "%d", &seed);
     
     fscanf(fp, "%s", str);
     assert(strcmp(str, "length") == 0);
     fscanf(fp, "%d", &length);

     fscanf(fp, "%s", str);
     assert(strcmp(str, "maxgen") == 0);
     fscanf(fp, "%d", &maxgen);
     
     fscanf(fp, "%s", str);
     assert(strcmp(str, "outputfile") == 0);
     fscanf(fp, "%s", outfile); /* fscanf() returns EOF if
                                   reading failed. */
     fscanf(fp, "%s", str);
     assert(strcmp(str, "mutation_type") == 0);
     fscanf(fp, "%d", &mutation_type);
    
     fscanf(fp, "%s", str);
     assert(strcmp(str, "recombination_type") == 0);
     fscanf(fp, "%d", &recombination_type);

     fscanf(fp, "%s", str);
     assert(strcmp(str, "mutation_probability") == 0);
     fscanf(fp, "%le", &mutat_prob);

     fscanf(fp, "%s", str);
     assert(strcmp(str, "recombination_probability") == 0);
     fscanf(fp, "%le", &recom_prob);

     fscanf(fp, "%s", str);
     assert(strcmp(str, "bit_turn_probability") == 0);
     result = fscanf(fp, "%le", &bit_turn_prob);


     assert(result != EOF); /* no EOF, outfile correctly read */
     
     srand(seed); /* seeding random number generator */

     fclose(fp);

     weights = (double *) malloc(length * dimension * sizeof(double));
     profits =  (double *) malloc(length * dimension * sizeof(double));
     profitSums = (double *) malloc(dimension * sizeof(double));
     capacities = (double *) malloc(dimension * sizeof(double));
     selectOrder = (int *) malloc(length * sizeof(int));
     
     /* --- DATA INJECTION (2026-07-26) -------------------------------------
        The original code GENERATED a random instance here:
            weights[..] = RandomInt(10,100); profits[..] = RandomInt(10,100);
            capacities[i] = 0.5 * weightSum;
        We instead READ the thesis's own Zitzler-Thiele instance from
        'pisa_instance.txt'. This changes ONLY where the problem data comes
        from -- exactly the role MOKPX.m plays for PlatEMO. Nothing else in
        this module changes: in particular the ORIGINAL greedy repair
        (profitWeightRatios / selectOrder / qsort, below) is untouched, and
        the SELECTOR (SPEA2 / NSGA2 / IBEA / HypE -- the actual algorithm) is
        a separate program that is not modified at all.

        File format (whitespace separated):
            <dimension> <length>
            weights  : dimension x length, row-major by objective
            profits  : dimension x length, row-major by objective
            capacities: dimension values
     -------------------------------------------------------------------- */
     {
	 FILE *inst;
	 int dimChk = -1, lenChk = -1;
	 int ii, jj;
	 inst = fopen("pisa_instance.txt", "r");
	 if (inst == NULL)
	 {
	     fprintf(stderr, "KNAPSACK: cannot open pisa_instance.txt\n");
	     return(1);
	 }
	 if (fscanf(inst, "%d %d", &dimChk, &lenChk) != 2)
	 {
	     fprintf(stderr, "KNAPSACK: bad header in pisa_instance.txt\n");
	     fclose(inst); return(1);
	 }
	 if (dimChk != dimension || lenChk != length)
	 {
	     fprintf(stderr,
		     "KNAPSACK: instance is %dx%d but PISA_cfg/param say %dx%d\n",
		     dimChk, lenChk, dimension, length);
	     fclose(inst); return(1);
	 }
	 for (ii = 0; ii < dimension; ii++)
	     for (jj = 0; jj < length; jj++)
		 if (fscanf(inst, "%le", &weights[ii * length + jj]) != 1)
		 { fprintf(stderr, "KNAPSACK: short weights\n"); fclose(inst); return(1); }
	 for (ii = 0; ii < dimension; ii++)
	     for (jj = 0; jj < length; jj++)
		 if (fscanf(inst, "%le", &profits[ii * length + jj]) != 1)
		 { fprintf(stderr, "KNAPSACK: short profits\n"); fclose(inst); return(1); }
	 for (ii = 0; ii < dimension; ii++)
	     if (fscanf(inst, "%le", &capacities[ii]) != 1)
	     { fprintf(stderr, "KNAPSACK: short capacities\n"); fclose(inst); return(1); }
	 fclose(inst);
     }

     /* determine item order regarding max{profit[i]/weight[i]}
	(per item over all knapsacks) using quicksort
     */
     profitWeightRatios = (double *) malloc(length * sizeof(double));
     for (i = 0; i < length; i++)
     {
	 double max = profits[i] / weights[i];
	 for (j = 1; j < dimension; j++)
	 {
	     double temp = profits[j * length + i] / weights[j * length + i];
	     if (temp > max)  max = temp;
	 }
	 profitWeightRatios[i] = max;
	 selectOrder[i] = i;
     }
     qsort(selectOrder, length, sizeof(int), cmpItems);
     free(profitWeightRatios);
     
     /* calculate total of profits per objective */
     for (i = 0; i < dimension; i++)
     {
	 profitSums[i] = 0;
	 for (j = 0; j < length; j++)
	 {
	     profitSums[i] += profits[i * length + j];
	 }
     }
     return (0);
}


/* Performs variation. */
int variate(int *selected, int *result_ids)
{
     int result, i, k;

     result = 1;

     /* copying all individuals from selected */
     for(i = 0; i < mu; i++)
     {
          result_ids[i] = 
               add_individual(copy_individual(get_individual(selected[i])));
          if(result_ids[i] == -1)
          {
               log_to_file(log_file, __FILE__, __LINE__,
                           "copying + adding failed");
               return (1);
          }
     }
 
     /* if odd number of individuals, last one is
        left as is */
     if((((double)mu/2) - (int)(mu/2)) != 0) k = mu - 1; 
     else k = mu;

     /* do recombination */
     for(i = 0; i < k; i+= 2)
     {  
          result = 1;
          if (drand(1) <= recom_prob)
          {
               result = 1;
               if (recombination_type == 1)
               {
                    result = one_point_crossover(get_individual(result_ids[i]),get_individual(result_ids[i + 1]));
               }
               else if (recombination_type == 2)
               {
                    result = uniform_crossover(get_individual(result_ids[i]), get_individual(result_ids[i + 1]));
               }
               else if (recombination_type == 0)
                    result = 0;

               if (result != 0)
                    log_to_file(log_file, __FILE__, 
                                __LINE__, "recombination failed!");
          }
     }
     
     /* do mutation */
     for(i = 0; i < mu; i++)
     {
          result = 1;
          if (drand(1) <= mutat_prob) /* only mutate with mut.probability */
          { 
               if(mutation_type == 1)
               {
                    result = one_bit_mutation(get_individual(result_ids[i]));
               }
               else if(mutation_type == 2)
               {
                    result = indep_bit_mutation(get_individual(result_ids[i]));
               }
               else if(mutation_type == 0)
               {
                    result = 0;
               }
    
               if(result_ids[0] == -1)
                    log_to_file(log_file, __FILE__, __LINE__,
                                "mutation failed!");
          }
     }
     
     /* do evaluation */
     for(i = 0; i < mu; i++)
     {
	 int result;
	 result = eval(get_individual(result_ids[i]));
     }
     
     return (0);
}


/* flip one bit at random position */
int one_bit_mutation(individual *ind)
{
     int position;

     if(ind == NULL)
          return (1);
     /* flip bit at position */
     position = irand(ind->length);
     
     if(ind->bit_string[position] == 0)
          ind->bit_string[position] = 1;
     else
          ind->bit_string[position] = 0;
  
     return (0);
}


/* flip all bits with certain probability */
int indep_bit_mutation(individual *ind)
{
     int i;
     double probability;
     /* absolute probability */
     probability = bit_turn_prob;
     if(ind == NULL)
          return (1);

     for(i = 0; i < ind->length; i++)
     {
          if(drand(1) < probability)
          {
               /* flip bit at position i*/
               if(ind->bit_string[i] == 0)
                    ind->bit_string[i] = 1;
               else
                    ind->bit_string[i] = 0;
          }
     }
     
     return (0);
}


/* do a one point crossover on ind1 and 2, the individual are
   overwritten! */
int one_point_crossover(individual *ind1, individual *ind2)
{
     int position, i;
     int *bit_string_ind2;
     bit_string_ind2 = (int *) malloc(sizeof(int) * ind2->length);
  
     for(i = 0; i < ind2->length; i++)
     {
          bit_string_ind2[i] = ind2->bit_string[i];
     }

     position = irand(ind2->length);

     for(i = 0; i < position; i++) {
          ind2->bit_string[i] = ind1->bit_string[i];
          ind1->bit_string[i] = bit_string_ind2[i];   
     }  

     free(bit_string_ind2);

     return(0);
}


/* do a uniform crossover on ind1 and 2, the individual are
   overwritten! */
int uniform_crossover(individual *ind1, individual *ind2)
{

     int choose, i;
     int *bit_string_ind2;
     bit_string_ind2 = (int *) malloc(sizeof(int) * ind2->length);
  
     for(i = 0; i < ind2->length; i++)
     {
          bit_string_ind2[i] = ind2->bit_string[i];
     }

     for(i = 0; i < ind2->length; i++) {
          choose = irand(2);
          if(choose == 1) /* switch around bits */
          { 
               ind2->bit_string[i] = ind1->bit_string[i];
               ind1->bit_string[i] = bit_string_ind2[i];
          } /* else leave bit as is */   
     }  

     free(bit_string_ind2);

     return (0);
}


/* Generate a random integer. */
int irand(int range)
{
     int j;
     j=(int) ((double) range * (double) rand() / (RAND_MAX + 1.0));
     return (j);
}


/* Generate a random double. */
double drand(double range)
{
     double j;
     j=(range * (double) rand() / (RAND_MAX + 1.0));
     return (j);
}


/* Determines the objective value based on KNAPSACK.
   In order to maximize the profits this function is minimized.
   PISA always minimizes. */
int eval(individual *ind)
{
/* knapsack problem
   calculate profits - abuse 'objectives' array for storing the used
   capacity per knapsack
*/
    int i;
    int j;
    int lastIndexUsed = -1;
    
    for (i = 0; i < dimension; i++)
	ind->objective_value[i] = 0;

    for (i = 0; i < length; i++)
    {
	int index = selectOrder[i];
	if (ind->bit_string[index] == 1)
	{
	    /* check whether item fits into knapsacks */
	    for (j = 0; j < dimension; j++)
		if (weights[j * length + index] +
		    ind->objective_value[j] > capacities[j]) 
		    break;
	    if (j == dimension)
	    {
		/* put item into knapsacks */
		int k;
		for (k = 0; k < dimension; k++)
		    ind->objective_value[k] += weights[k * length + index];
		lastIndexUsed = i;
	    }
	    else
		/* item too big -> do not examine remaining items */
		break;
	}
    }
    /* store profits actually achieved in 'ind->objective_value' array */
    for (j = 0; j < dimension; j++)
	ind->objective_value[j] = 0;
    
    while (i > 0)
    {
	int index;
	i--;
	index = selectOrder[i];
	if (ind->bit_string[index] == 1)
            for (j = 0; j < dimension; j++)
                ind->objective_value[j] += profits[j * length + index];
    }

    /* for minimization */
    for (j = 0; j < dimension; j++)
	ind->objective_value[j] = profitSums[j] - ind->objective_value[j];
    return (0);
}


/* create a random new individual and allocate memory for it,
   returns a pointer to the new individual */
individual *new_individual()
{
     individual *return_ind;
     int i;
     int result;

     return_ind = (individual *) malloc(sizeof(individual));
     return_ind->bit_string = (int *) malloc(sizeof(int) * length);
     return_ind->objective_value = (double *) malloc(sizeof(double) *
                                                     dimension);
 
     for (i = 0; i < length; i++)
          return_ind->bit_string[i] = irand(2);

     return_ind->length = length;

     /* evaluating the objective functions */
     result = eval(return_ind);

     return (return_ind);
}


/* copy an individual and return the pointer to it */
individual *copy_individual(individual *ind)
{
     individual *return_ind;
     int i;

     return_ind = (individual *) malloc(sizeof(individual));
     return_ind->bit_string = (int *) malloc(sizeof(int) * length);
     return_ind->objective_value = (double *) malloc(sizeof(double) *
                                                     dimension);
     
     for (i = 0; i < length; i++)
          return_ind->bit_string[i] = ind->bit_string[i];

     for (i = 0; i < dimension; i++)
	  return_ind->objective_value[i] = ind->objective_value[i];

     return_ind->length = ind->length;

     return(return_ind);
}

/* Writes the index, objective values and bit string of
   all individuals in global_population to 'out_filename'. */
void write_output_file()
{
     int j, current_id;
     FILE *fp_out;
     individual *temp;
     
     fp_out = fopen(outfile, "w");
     assert(fp_out != NULL);

     current_id = get_first();

     while (current_id != -1)
     {       
	  temp = get_individual(current_id);
          fprintf(fp_out, "%d ", current_id); /* write index */
	  for (j = 0; j < dimension; j++)
	  {
	      fprintf(fp_out, "%d ", (int)(profitSums[j]
		      - get_objective_value(current_id, j)));
	  }
          
	  for (j = 0; j < temp->length; j++)
          {
               fprintf(fp_out, "%d", temp->bit_string[j]);
          }
          fprintf(fp_out, "\n");
	  current_id = get_next(current_id);
     }

     fclose(fp_out);
}


/**********| addition for KNAPSACK end |*******/
