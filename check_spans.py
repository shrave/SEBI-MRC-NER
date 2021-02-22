import json

#catch all the same labels and put into json.
# print(len(data))



def convert_file(input_file, output_file, tag2query_file):
	"""
	Convert MSRA raw data to MRC format
	"""
	count = 0
	tot = 0
	origin_count = 0
	new_count = 0
	tag2query = json.load(open(tag2query_file))
	mrc_samples = []
	with open("result.json", "r") as read_file:
		data = json.load(read_file)
	for rule in data:
		origin_count += 1
		context = rule['data']['text']
		context = context.rstrip('\n				 ')
		context = context.strip()
		span_token_converted = {}
		span_token_convert = {}
		#config_bert = '/home2/shravya.k/SEBI-MRC-NER/config/en_bert_base_uncased.json'
		#tokenizer = BertTokenizer4Tagger.from_pretrained(config_bert, do_lower_case=True)
		#tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", use_lower=True)
		tokens_list = context.split()
		#tokens_list = tokenizer.tokenize(context)
		#tokens_list = tokens_list[1:-1]
		token_pointer = 0
		char_pointer = 0
		temp_word = ''
		prev_pointer = 0
		m = context.split(' ')
		context = " ".join(m)
		context = context.strip()
		for cr in range(0,len(context)):
#			print("context_car ->"+context[cr])
			k = tokens_list[token_pointer].replace("#", "")
#			print("current_token ->"+k)
#			if context[cr] == ' ':
#				prev_pointer += 1
			if context[cr] == k:
				span_token_converted[(cr,cr)] = (token_pointer+1, tokens_list[token_pointer])
				span_token_convert[cr] = token_pointer+1
#				print(k)
#				print(context[cr])
#				print('single char caught')
				token_pointer += 1
				prev_pointer = cr
			else:
				temp_word += context[cr]
				#temp_word.strip()
#				print("temp word ->"+temp_word)
				if temp_word.strip() == k:
					span_token_converted[(cr-len(temp_word.strip())+1,cr)] = (token_pointer+1, tokens_list[token_pointer])
					span_token_convert[cr-len(temp_word.strip())+1] = token_pointer
#					print(k)
#					print(temp_word)
#					print('passed')
					token_pointer +=1
					prev_pointer = cr
					temp_word = ''
#		print(span_token_converted)
#		print(context)
#		print(tokens_list)
#		print(len(tokens_list))
#		break
#		count = 0
#		tot = 0 
		start_span_by_labels = {}
		end_span_by_labels = {}
		start_end_span_labels = {}
		for ents in rule['completions']:
			# print(len(ents))
			for ent in ents['result']:
				start_token = span_token_convert[ent['value']['start']] if ent['value']['start'] in span_token_convert else span_token_convert[min(span_token_convert.keys(), key=lambda k: abs(k-ent['value']['start']))]
				end_token = span_token_convert[ent['value']['end']] if ent['value']['end'] in span_token_convert else span_token_convert[min(span_token_convert.keys(), key=lambda k: abs(k-ent['value']['end']))]
#				print('character token')
				act = ent['value']['text']
				pred = (' ').join(tokens_list[start_token: end_token]).strip()
				#count = 0
				if act in pred:
					count += 1
				tot += 1
				print(ent['value']['text'])
#				print('converted token')
				print((' ').join(tokens_list[start_token: end_token]).strip())
	print(count/float(tot))
def main():
	# sebi_raw_rules = ""
	# sebi_mrc_dir = "/mnt/mrc/genia/genia_raw/mrc_format"
	tag2query_file = "../ner2mrc/queries/sebi_entities.json"	 
	# genia_raw_dir = "/mnt/mrc/genia/genia_raw"
	# genia_mrc_dir = "/mnt/mrc/genia/genia_raw/mrc_format"
	# tag2query_file = "queries/genia.json"
	# os.makedirs(sebi_mrc_dir, exist_ok=True)
	for phase in ["train", "dev", "test"]:
		# old_file = os.path.join(genia_raw_dir, f"{phase}.genia.json")
		old_file = 'sebi_annotated.json'
		new_file = f"mrc-ner.{phase}"
		convert_file(old_file, new_file, tag2query_file)


if __name__ == '__main__':
	main()


