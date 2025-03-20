import logging
import os
import time
from urllib.parse import urlparse

import pkg.initializers.utils.utils as utils
from pkg.initializers.dataset.cache_initalizer import deploy_lws_with_substitution
from pkg.initializers.dataset.huggingface import HuggingFace

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    level=logging.INFO,
)


def main():
    logging.info("Starting dataset initialization")

    # try:
    #     storage_uri = os.environ[utils.STORAGE_URI_ENV]
    # except Exception as e:
    #     logging.error("STORAGE_URI env variable must be set.")
    #     raise e
    #
    # match urlparse(storage_uri).scheme:
    #     # TODO (andreyvelich): Implement more dataset providers.
    #     case utils.HF_SCHEME:
    #         hf = HuggingFace()
    #         hf.load_config()
    #         hf.download_dataset()
    #     case _:
    #         logging.error("STORAGE_URI must have the valid dataset provider")
    #         raise Exception
    try:
        deploy_lws_with_substitution(
            'pkg/initializers/dataset/cache-initializer-template.yaml',
            namespace='cache-test',
            substitutions={
                'NAME': 'cache-lws-test',
                'IAM_ROLE': 'arn:aws:iam::533547146520:role/kubeflow-infra-summit',
                'SIZE': '3',
                'IMAGE': 'docker.apple.com/achitneni/arrow_cache:b59025d',

            },
            timeout=600
        )
        time.sleep(30)
    except Exception as e:
        logging.error("error creating cache-lws")
        raise e


if __name__ == "__main__":
    main()
