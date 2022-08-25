filename='parameter_alpha.txt'
n=1
while read line; do

	filename2='parameter_margin.txt'
	o=1
	while read line2; do
		echo $line2
		o=$((o+1))
	done < $filename2

	# reading each line
	echo $line
	n=$((n+1))
done < $filename
