for i in {1..20}
do 
sbatch jobs/job_${i}.slurm 
done
