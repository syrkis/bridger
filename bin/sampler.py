# sampler.py
#	constructs sample data sets
# by: Group 2

# imports
import os
import reader
import json

domains = os.listdir(reader.reviews_path)
for domain in domains:
	target_file = f'data/samples/{domain.split(".")[0]}.json'
	data = []
	generator = reader.sample_domain(domain)
	for i in range(2000):
		datum = generator.__next__()
		data.append(datum)

	with open(target_file, 'w') as f:
		f.write(json.dumps(data))

print('DONE')
