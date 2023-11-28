# Fine-tuning Llama2-ko-7b 4-bit Quantization QLoRA  w/ KorQuAD_2.0

한국어 사전학습 모델 [Llama2-ko-7b](https://huggingface.co/beomi/llama-2-ko-7b)를 한국어 ML Comprehenzsion 데이터셋, [KorQuAD 2.0](https://korquad.github.io/)로 4bit QLoRA를 적용해서 
Fine-tuning 하는 코드입니다. Fine-tuning 결과 체크포인트는 [여기서](https://huggingface.co/leeseeun/llama2-7b-ko-finetuning) 확인할 수 있습니다.

<br>

## 🐚 How to Fine-tuning Llama2-ko-7b

    python3 finetuning.py

<br>

- 해당 코드는 이준범님의 [Llama2-7b 한국어 사전학습 ver.](https://huggingface.co/beomi/llama-2-ko-7b)을 [KorQuAD_2.0 데이터셋](https://huggingface.co/datasets/leeseeun/KorQuAD_2.0) 2,000개를 이용해서 fine-tuning 하는 코드입니다.
- 해당 코드를 실행하기 전 결과가 저장될 로컬경로([`output_dir`](https://github.com/SEUNKOREA/Llama2-7b-ko-FT/blob/3aa6ab0c388c924e975d101c7b368a0b52d815f0/finetuning.py#L109),[`output_merged_dir`](https://github.com/SEUNKOREA/Llama2-7b-ko-FT/blob/3aa6ab0c388c924e975d101c7b368a0b52d815f0/finetuning.py#L117C5-L117C22)), [허깅페이스 리포지토리 주소](https://github.com/SEUNKOREA/Llama2-7b-ko-FT/blob/3aa6ab0c388c924e975d101c7b368a0b52d815f0/finetuning.py#L127)가 올바른지 확인하세요.
- [사용한 KorQuAD_2.0 데이터셋](https://huggingface.co/datasets/leeseeun/KorQuAD_2.0)은 [해당 링크](https://github.com/korquad/korquad.github.io/tree/master/dataset/KorQuAD_2.1/train)에 있는 zip파일을 모두 합친 후 json 파일로 변환하여 허깅페이스 포맷으로 변환 후 사용하였습니다.
    - zip파일을 모두 합친 json 파일은 [여기서](https://github.com/SEUNKOREA/Llama2-7b-ko-FT/blob/main/newfile.json) 확인할 수 있습니다.
    - 데이터셋을 만드는 방법은 아래의 ["Create 🤗 datasets from KorQuAD"](https://github.com/SEUNKOREA/Llama2-7b-ko-FT/tree/main#-create--datasets-from-korquad) 가이드를 참고하세요.
- 모델의 양자화 관련 설정값은 [여기서](https://github.com/SEUNKOREA/Llama2-7b-ko-FT/blob/3aa6ab0c388c924e975d101c7b368a0b52d815f0/model_utils.py#L38) 확인할 수 있습니다.
    - 각 파라미터 설정값에 대한 설명은 [공식문서](https://huggingface.co/docs/transformers/main_classes/quantization#transformers.BitsAndBytesConfig)를 참고하세요.
- PEFT 관련 설정값은 [여기서](https://github.com/SEUNKOREA/Llama2-7b-ko-FT/blob/3aa6ab0c388c924e975d101c7b368a0b52d815f0/model_utils.py#L47) 확인할 수 있습니다.
    - 각 파라미터 설정값에 대한 설명은 [공식문서](https://huggingface.co/docs/peft/main/en/package_reference/tuners#peft.LoraConfig)를 참고하세요.
- 훈련 관련 설정값은 [여기서](https://github.com/SEUNKOREA/Llama2-7b-ko-FT/blob/3aa6ab0c388c924e975d101c7b368a0b52d815f0/finetuning.py#L28C13-L28C13) 확인할 수 있습니다.
    - 각 파라미터 설정값에 대한 설명은 [공식문서](https://huggingface.co/docs/transformers/v4.35.2/en/main_classes/trainer#transformers.TrainingArguments)를 참고하세요.
- fine-tuning에 사용할 데이터의 개수를 변경하고 싶은 경우, [해당 코드](https://github.com/SEUNKOREA/Llama2-7b-ko-FT/blob/3aa6ab0c388c924e975d101c7b368a0b52d815f0/finetuning.py#L97C1-L97C1)의 `split`을 수정하여 변경할 수 있습니다.
    - 자세한 변경 가이드는 [허깅페이스의 공식문서](https://huggingface.co/docs/datasets/v1.11.0/splits.html#slicing-api)를 참조하세요.
- 데이터셋의 청크사이즈를 변경하고 싶은 경우, [해당 코드](https://github.com/SEUNKOREA/Llama2-7b-ko-FT/blob/3aa6ab0c388c924e975d101c7b368a0b52d815f0/finetuning.py#L102)의 `max_length`을 수정하여 변경할 수 있습니다.
    - 기본 값은 모델이 처리할 수 있는 최대 토큰의 길이를 사용하도록 되어있습니다.



<br>

## 🧁 Generate QA

    python3 generation.py


- 해당 코드는 한국어 사전학습 모델 [Llama2-ko-7b](https://huggingface.co/beomi/llama-2-ko-7b)를 한국어 ML Comprehenzsion 데이터셋, [KorQuAD 2.0](https://korquad.github.io/)로 4bit QLoRA를 적용해서 Fine-tuning한 모델의 [체크포인트](https://huggingface.co/leeseeun/llama2-7b-ko-finetuning)를 불러와서 "인공지능에 대해서 설명해줘."라는 질문에 대해 답변을 생성하는 코드입니다.
- 질문의 내용을 변경하고 싶은 경우, 코드의 [`eval_data`](https://github.com/SEUNKOREA/Llama2-7b-ko-FT/blob/3aa6ab0c388c924e975d101c7b368a0b52d815f0/generation.py#L21C14-L21C29)를 수정하면 됩니다.
- 다른 리포지토리 경로로 바꿀 경우, 예상치 못한 결과가 발생할 수 있습니다.


<br>

## 🎲 Create 🤗 datasets from KorQuAD

    python3 create_dataset.py

- 해당 [사용한 KorQuAD_2.0 데이터셋](https://huggingface.co/datasets/leeseeun/KorQuAD_2.0)은 [해당 링크](https://github.com/korquad/korquad.github.io/tree/master/dataset/KorQuAD_2.1/train)에 있는 zip파일을 모두 합친 후 json 파일을 전처리 후 허깅페이스 포맷으로 변환 후 허깅페이스 허브에 업로드 하는 코드입니다.
- zip파일을 모두 합친 json 파일은 [여기서](https://github.com/SEUNKOREA/Llama2-7b-ko-FT/blob/main/newfile.json) 확인할 수 있습니다.
- 해당 코드를 실행하기 전, json 원본 데이터 경로([`data_path`](https://github.com/SEUNKOREA/Llama2-7b-ko-FT/blob/abfbb6a9b7b54edc6fa0dc1f3bc467b4adcbc7f1/create_dataset.py#L7C5-L7C14))와 업로드할 [허깅페이스 리포지토리 주소](https://github.com/SEUNKOREA/Llama2-7b-ko-FT/blob/abfbb6a9b7b54edc6fa0dc1f3bc467b4adcbc7f1/create_dataset.py#L34)가 올바른지 확인하세요.

<br>

## Acknowlegemnets
해당 코드를 테스트하고 실행함에 있어서 [(주)딥로딩](https://www.deeploading.com/)으로부터 서버를 지원받았습니다.
<br>
유용한 지원을 해주신 [(주)딥로딩](https://www.deeploading.com/)에 감사의 인사를 전합니다.
