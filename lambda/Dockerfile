FROM public.ecr.aws/lambda/python:3.10

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install -r requirements.txt
RUN pip install transformers

# Download NLTK punkt tokenizer
COPY utils.py ${LAMBDA_TASK_ROOT}
RUN python ${LAMBDA_TASK_ROOT}/utils.py
ENV NLTK_DATA=${LAMBDA_TASK_ROOT}

COPY model.py ${LAMBDA_TASK_ROOT}
COPY lambda_function.py ${LAMBDA_TASK_ROOT}
CMD [ "lambda_function.split_file" ]