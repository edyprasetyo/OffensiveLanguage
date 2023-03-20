from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import psutil

pid = psutil.Process()
initial_cpu = pid.cpu_percent()
initial_ram = pid.memory_info().rss / 1024 / 1024


pretrained = "mdhugol/indonesia-bert-sentiment-classification"

model = AutoModelForSequenceClassification.from_pretrained(pretrained)
tokenizer = AutoTokenizer.from_pretrained(pretrained)

sentiment_analysis = pipeline(
    "sentiment-analysis", model=model, tokenizer=tokenizer)

label_index = {'LABEL_0': 'positive',
               'LABEL_1': 'neutral', 'LABEL_2': 'negative'}

my_list = [
    'kemarin gue datang ke tempat makan baru yang ada di dago atas . gue kira makanan nya enak karena harga nya mahal . ternyata , boro-boro . tidak mau lagi deh ke tempat itu . sudah mana tempat nya juga tidak nyaman banget , terlalu sempit .',
]

for str in my_list:
    result = sentiment_analysis(str)
    status = label_index[result[0]['label']]
    score = result[0]['score']
    print(f'Text: {str} | Label : {status} ({score * 100:.1f}%)')

final_cpu = pid.cpu_percent()
final_ram = pid.memory_info().rss / 1024 / 1024

print(f"CPU usage: {initial_cpu - initial_cpu}%")
print(f"RAM usage: {initial_ram - initial_ram} MB")

print(f"CPU usage: {final_cpu - initial_cpu}%")
print(f"RAM usage: {final_ram - initial_ram} MB")
