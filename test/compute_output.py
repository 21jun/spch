import jiwer
import json

CER = []
WER = []
with open("test/100_librispeech-dev-clean.txt", "r") as f:
    lines = f.readlines()
#remove the last line
lines = lines[:-1]
data = [json.loads(line) for line in lines]
# print(data[0])


for datum in data:
    ref = datum["reference"].strip()
    hyp = datum["prediction"].strip()
    cer = jiwer.cer(ref, hyp)
    wer= jiwer.wer(ref, hyp)


    CER.append(cer)
    WER.append(wer)

    if wer > 0.3:
        print("Reference: ", ref)
        print("Prediction: ", hyp)
        print("CER: ", cer)
        print("WER: ", wer)
        print("")


print("Average CER: ", sum(CER)/len(CER))

print("Average WER: ", sum(WER)/len(WER))
