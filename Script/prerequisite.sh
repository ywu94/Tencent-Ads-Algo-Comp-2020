CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$CWD")"

TRAIN_SET_PATH="https://tesla-ap-shanghai-1256322946.cos.ap-shanghai.myqcloud.com/cephfs/tesla_common/deeplearning/dataset/algo_contest/train_preliminary.zip"
TEST_SET_PATH="https://tesla-ap-shanghai-1256322946.cos.ap-shanghai.myqcloud.com/cephfs/tesla_common/deeplearning/dataset/algo_contest/test.zip"


TRAIN_FILE_DIR="${CWD}/train_artifact"
if [ -d "${TRAIN_FILE_DIR}" ] && [ -f "${TRAIN_FILE_DIR}/user.csv" ] && [ -f "${TRAIN_FILE_DIR}/ad.csv" ] && [ -f "${TRAIN_FILE_DIR}/click_log.csv" ]
then
    echo "Train artifact directory and training files exist"
else
    echo "Missing train artifact directory, recreating..."
    rm -rf ${TRAIN_FILE_DIR}
    mkdir -p ${TRAIN_FILE_DIR}
    if [ -f "${CWD}/train_preliminary.zip" ]
    then
    	:
    else
    	wget ${TRAIN_SET_PATH} -O ${CWD}/train_preliminary.zip
    fi
    unzip ${CWD}/train_preliminary.zip -d ${CWD}
    mv  -v ${CWD}/train_preliminary/* ${TRAIN_FILE_DIR}
    rm ${CWD}/train_preliminary.zip
    rm -rf ${CWD}/train_preliminary
    echo "Train artifact directory and training files ready"
fi

TEST_FILE_DIR="${CWD}/test_artifact"
if [ -d "${TEST_FILE_DIR}" ] && [ -f "${TEST_FILE_DIR}/ad.csv" ] && [ -f "${TEST_FILE_DIR}/click_log.csv" ]
then
    echo "Test artifact directory and training files exist"
else
    echo "Missing test artifact directory, recreating..."
    rm -rf ${TEST_FILE_DIR}
    mkdir -p ${TEST_FILE_DIR}
    if [ -f "${CWD}/test.zip" ]
    then
    	:
    else
    	wget ${TEST_SET_PATH} -O ${CWD}/test.zip
    fi
    unzip ${CWD}/test.zip -d ${CWD}
    mv  -v ${CWD}/test/* ${TEST_FILE_DIR}
    rm ${CWD}/test.zip
    rm -rf ${CWD}/test
    echo "Test artifact directory and training files ready"
fi

INPUT_DIR="${CWD}/input_artifact"
if [ -d "${INPUT_DIR}" ]
then 
	echo "Input directory exists"
else
	echo "Missing input directory, recreating..."
	mkdir -p ${INPUT_DIR}
	echo "Input directory ready"
fi

EMBED_DIR="${CWD}/embed_artifact"
if [ -d "${EMBED_DIR}" ]
then 
	echo "Embed directory exists"
else
	echo "Missing embed directory, recreating..."
	mkdir -p ${EMBED_DIR}
	echo "Embed directory ready"
fi

MODEL_DIR="${CWD}/model_artifact"
if [ -d "${MODEL_DIR}" ]
then 
	echo "Model directory exists"
else
	echo "Missing model directory, recreating..."
	mkdir -p ${MODEL_DIR}
	echo "Model directory ready"
fi

OUTPUT_DIR="${CWD}/output_artifact"
if [ -d "${OUTPUT_DIR}" ]
then 
	echo "Output directory exists"
else
	echo "Missing output directory, recreating..."
	mkdir -p ${OUTPUT_DIR}
	echo "Output directory ready"
fi

