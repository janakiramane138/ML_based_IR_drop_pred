i=0
for h in benchmarks/testcase*; do
  i=$((i + 1))
  echo "python3 ir_solver_sparse.py --input_file $h --output_file phase11_output/testcase${i}_out.v"
done

