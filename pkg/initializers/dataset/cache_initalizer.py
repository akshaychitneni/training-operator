import os
from typing import Optional

import yaml
import time
from kubernetes import client, config, utils
from kubernetes.client import V1OwnerReference
from kubernetes.client.rest import ApiException
from kubernetes.utils import FailToCreateError


def deploy_lws_with_substitution(train_job_name, yaml_path, config_file: Optional[str] = None, namespace='default', substitutions=None,
                                 timeout=300):
    """
    Deploys a parameterized LeaderWorkerSet YAML with ServiceAccount, environment substitution,
    and waits for deployment readiness.

    Args:
        train_job_name(str): train_job_name
        yaml_path (str): Path to YAML file containing configuration
        namespace (str): Target Kubernetes namespace
        substitutions (dict): Additional variables for substitution
        timeout (int): Maximum wait time in seconds

    Returns:
        bool: True if deployment succeeded and became ready
    """
    with open(yaml_path, 'r') as f:
        content = f.read()

    sub_vars = {**os.environ, **(substitutions or {})}

    for key, value in sub_vars.items():
        content = content.replace(f'${key}', value)
        content = content.replace(f'${{{key}}}', value)

    resources = list(yaml.safe_load_all(content))

    if config_file or not is_running_in_k8s():
        config.load_kube_config(config_file=config_file)
    else:
        config.load_incluster_config()

    api_client = client.ApiClient()
    core_v1 = client.CoreV1Api(api_client)
    custom_api = client.CustomObjectsApi(api_client)
    training_job = custom_api.get_namespaced_custom_object(
        group="trainer.kubeflow.org",
        version="v1alpha1",
        plural="trainjobs",
        namespace=namespace,
        name=train_job_name
    )

    # Create owner reference from TrainingJob
    owner_ref = V1OwnerReference(
        api_version="trainer.kubeflow.org/v1alpha1",
        kind="TrainJob",
        name=training_job["metadata"]["name"],
        uid=training_job["metadata"]["uid"],
        block_owner_deletion=False
    )

    created_lws = []
    created_sa = []

    try:
        for resource in resources:
            # if resource['kind'] == 'ServiceAccount':
            #     metadata = resource['metadata']
            #     sa_name = metadata['name']
            #
            #     try:
            #         core_v1.read_namespaced_service_account(
            #             name=sa_name,
            #             namespace=namespace
            #         )
            #     except ApiException as e:
            #         if e.status == 404:
            #             core_v1.create_namespaced_service_account(
            #                 namespace=namespace,
            #                 body=resource
            #             )
            #             created_sa.append((sa_name, namespace))
            #     except Exception as e:
            #         print(f"Error checking ServiceAccount: {e}")
            #         raise
            resource['metadata']['owner_references'] = [owner_ref]
            if resource['kind'] == 'LeaderWorkerSet':
                created_lws.append(resource)
            else:
                try:
                    utils.create_from_dict(api_client, resource, namespace=namespace)
                except FailToCreateError as ex:
                    for e in ex.api_exceptions:
                        if e.status == 409:
                            print(
                                f"Resource {resource['kind']}/{resource['metadata']['name']} already exists, skipping creation")
                        else:
                            raise

        lws_to_watch = []
        for lws_resource in created_lws:
            group = 'leaderworkerset.x-k8s.io'
            version = 'v1'
            plural = 'leaderworkersets'
            name = lws_resource['metadata']['name']

            custom_api.create_namespaced_custom_object(
                group,
                version,
                namespace,
                plural,
                lws_resource,
            )
            lws_to_watch.append((group, version, plural, name))

        start_time = time.time()
        for group, version, plural, name in lws_to_watch:
            while time.time() - start_time < timeout:
                try:
                    lws = custom_api.get_namespaced_custom_object(
                        group=group,
                        version=version,
                        plural=plural,
                        name=name,
                        namespace=namespace
                    )

                    if any(c['type'] == 'Available' and c['status'] == 'True'
                           for c in lws.get('status', {}).get('conditions', [])):
                        break

                    time.sleep(5)
                except ApiException:
                    time.sleep(2)
            else:
                raise TimeoutError(f"LWS {name} didn't become ready in {timeout}s")

        return True

    except ApiException as e:
        print(f"Deployment failed: {e.reason}")
        for sa_name, ns in created_sa:
            try:
                core_v1.delete_namespaced_service_account(
                    name=sa_name,
                    namespace=ns
                )
            except Exception as cleanup_error:
                print(f"Error cleaning up SA {sa_name}: {cleanup_error}")
        return False


if __name__ == "__main__":
    deploy_lws_with_substitution(
        'cache-initializer-template.yaml',
        namespace='cache-test',
        substitutions={
            'NAME': 'cache-lws-test',
            'IAM_ROLE': 'arn:aws:iam::533547146520:role/kubeflow-infra-summit',
            'SIZE': '3',
            'IMAGE': 'docker.apple.com/achitneni/arrow_cache:b59025d',

        },
        timeout=600
    )
    # from kubernetes.client import ApisApi
    # print(ApisApi().get_api_versions().groups)


def is_running_in_k8s() -> bool:
    return os.path.isdir("/var/run/secrets/kubernetes.io/")
