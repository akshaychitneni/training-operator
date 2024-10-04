/*
Copyright 2024 The Kubeflow Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package webhookv2

import (
	"context"
	"fmt"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/klog/v2"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"slices"
	"strconv"

	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/webhook"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"

	kubeflowv2 "github.com/kubeflow/training-operator/pkg/apis/kubeflow.org/v2alpha1"
)

var (
	specPath                = field.NewPath("spec")
	trainingRuntimeRefPath  = specPath.Child("trainingRuntimeRef")
	trainingRuntimeNamePath = trainingRuntimeRefPath.Child("name")
)

type TrainJobWebhook struct {
	Client client.Client
}

func setupWebhookForTrainJob(mgr ctrl.Manager) error {
	webhook := &TrainJobWebhook{
		Client: mgr.GetClient(),
	}
	return ctrl.NewWebhookManagedBy(mgr).
		For(&kubeflowv2.TrainJob{}).
		WithValidator(webhook).
		Complete()
}

// +kubebuilder:webhook:path=/validate-kubeflow-org-v2alpha1-trainjob,mutating=false,failurePolicy=fail,sideEffects=None,groups=kubeflow.org,resources=trainjobs,verbs=create;update,versions=v2alpha1,name=validator.trainjob.kubeflow.org,admissionReviewVersions=v1

var _ webhook.CustomValidator = (*TrainJobWebhook)(nil)

func (w *TrainJobWebhook) ValidateCreate(ctx context.Context, obj runtime.Object) (admission.Warnings, error) {
	trainJob := obj.(*kubeflowv2.TrainJob)
	log := ctrl.LoggerFrom(ctx).WithName("trainJob-webhook")
	log.V(5).Info("Validating create", "TrainJob", klog.KObj(trainJob))
	return nil, validateTrainJob(ctx, w.Client, nil, trainJob).ToAggregate()
}

func (w *TrainJobWebhook) ValidateUpdate(ctx context.Context, old runtime.Object, new runtime.Object) (admission.Warnings, error) {
	oldTrainJob := old.(*kubeflowv2.TrainJob)
	newTrainJob := new.(*kubeflowv2.TrainJob)
	log := ctrl.LoggerFrom(ctx).WithName("trainJob-webhook")
	log.V(5).Info("Validating update", "TrainJob", klog.KObj(newTrainJob))
	return nil, validateTrainJob(ctx, w.Client, oldTrainJob, newTrainJob).ToAggregate()
}

func (w *TrainJobWebhook) ValidateDelete(context.Context, runtime.Object) (admission.Warnings, error) {
	return nil, nil
}

func validateTrainJob(ctx context.Context, client client.Client, oldJob, newJob *kubeflowv2.TrainJob) field.ErrorList {
	var allErrs field.ErrorList
	var trainingRuntime kubeflowv2.TrainingRuntime
	trainingRuntime.SetGroupVersionKind(schema.GroupVersionKind{
		Group:   *newJob.Spec.TrainingRuntimeRef.APIGroup,
		Version: "v1alpha1",
		Kind:    *newJob.Spec.TrainingRuntimeRef.Kind,
	})
	var key types.NamespacedName
	if *newJob.Spec.TrainingRuntimeRef.Kind == "ClusterTrainingRuntime" {
		key = types.NamespacedName{
			Name: newJob.Spec.TrainingRuntimeRef.Name,
		}
	} else {
		key = types.NamespacedName{
			Name:      newJob.Spec.TrainingRuntimeRef.Name,
			Namespace: newJob.Namespace,
		}
	}
	if err := client.Get(ctx, key, &trainingRuntime); err != nil {
		if errors.IsNotFound(err) {
			allErrs = append(allErrs, field.NotFound(trainingRuntimeNamePath, newJob.Spec.TrainingRuntimeRef.Name))
		} else {
			allErrs = append(allErrs, field.InternalError(trainingRuntimeNamePath, err))
		}
		return allErrs
	}

	allErrs = append(allErrs, validateTrainJobSpec(&newJob.Spec, &trainingRuntime)...)
	return allErrs
}

func validateTrainJobSpec(spec *kubeflowv2.TrainJobSpec, trainingRuntime *kubeflowv2.TrainingRuntime) field.ErrorList {
	var allErrs field.ErrorList
	if spec.Trainer != nil {
		numProcPerNodePath := specPath.Child("trainer").Child("numProcPerNode")
		if trainingRuntime.Spec.MLPolicy.MPI != nil {
			if _, err := strconv.Atoi(*spec.Trainer.NumProcPerNode); err != nil {
				allErrs = append(allErrs, field.Invalid(numProcPerNodePath, spec.Trainer.NumProcPerNode, "should have an int value"))
			}
		} else if trainingRuntime.Spec.MLPolicy.Torch != nil {
			allowedStringValList := []string{"auto", "cpu", "gpu"}
			if !slices.Contains(allowedStringValList, *spec.Trainer.NumProcPerNode) {
				if _, err := strconv.Atoi(*spec.Trainer.NumProcPerNode); err != nil {
					allErrs = append(allErrs, field.Invalid(numProcPerNodePath, spec.Trainer.NumProcPerNode, "should have an int value or auto/cpu/gpu"))
				}
			}
		}
	}

	if spec.ModelConfig != nil {
		if spec.ModelConfig.Input != nil {
			modelConfigInputPath := specPath.Child("modelConfig").Child("input")
			if len(trainingRuntime.Spec.Template.Spec.ReplicatedJobs) == 0 {
				allErrs = append(allErrs, field.Invalid(modelConfigInputPath, spec.ModelConfig.Input, "trainingRuntime should have replicated jobs configured with model config input set"))
			} else {
				initializerJobFound := false
				modelInitializerContainerFound := false
				for _, job := range trainingRuntime.Spec.Template.Spec.ReplicatedJobs {
					if job.Name == "Initializer" {
						initializerJobFound = true
						for _, container := range job.Template.Spec.Template.Spec.Containers {
							if container.Name == "model-initializer" {
								modelInitializerContainerFound = true
							}
						}
					}
				}
				if !initializerJobFound {
					allErrs = append(allErrs, field.Invalid(modelConfigInputPath, spec.ModelConfig.Input, "trainingRuntime should have replicated job configured with name - Initializer"))
				} else if !modelInitializerContainerFound {
					allErrs = append(allErrs, field.Invalid(modelConfigInputPath, spec.ModelConfig.Input, "trainingRuntime with replicated job initializer should have container with name - model-initializer"))
				}
			}
		}
		if spec.ModelConfig.Output != nil {
			modelConfigInputPath := specPath.Child("modelConfig").Child("output")
			if len(trainingRuntime.Spec.Template.Spec.ReplicatedJobs) == 0 {
				allErrs = append(allErrs, field.Invalid(modelConfigInputPath, spec.ModelConfig.Output, "trainingRuntime should have replicated jobs configured with model config output set"))
			} else {
				exporterJobFound := false
				modelExporterContainerFound := false
				for _, job := range trainingRuntime.Spec.Template.Spec.ReplicatedJobs {
					if job.Name == "Exporter" {
						exporterJobFound = true
						for _, container := range job.Template.Spec.Template.Spec.Containers {
							if container.Name == "model-exporter" {
								modelExporterContainerFound = true
							}
						}
					}
				}
				if !exporterJobFound {
					allErrs = append(allErrs, field.Invalid(modelConfigInputPath, spec.ModelConfig.Input, "trainingRuntime should have replicated job configured with name - Exporter"))
				} else if !modelExporterContainerFound {
					allErrs = append(allErrs, field.Invalid(modelConfigInputPath, spec.ModelConfig.Input, "trainingRuntime with replicated job initializer should have contianer with name - model-exporter"))
				}
			}
		}
	}

	if len(spec.PodSpecOverrides) != 0 {
		podSpecOverridesPath := specPath.Child("podSpecOverrides")
		jobsMap := map[string]bool{}
		for _, job := range trainingRuntime.Spec.Template.Spec.ReplicatedJobs {
			jobsMap[job.Name] = true
		}
		for idx, override := range spec.PodSpecOverrides {
			for _, job := range override.TargetReplicatedJobs {
				if ok, _ := jobsMap[job]; !ok {
					allErrs = append(allErrs, field.Invalid(podSpecOverridesPath, spec.PodSpecOverrides, fmt.Sprintf("job: %s, configured in the podOverride should be present in the referenced training runtime", job)))
				}
			}
			if len(override.Containers) != 0 {
				containerMap := map[string]bool{}
				for _, job := range trainingRuntime.Spec.Template.Spec.ReplicatedJobs {
					for _, container := range job.Template.Spec.Template.Spec.Containers {
						containerMap[container.Name] = true
					}
				}
				containerOverridePath := podSpecOverridesPath.Index(idx)
				for _, container := range override.Containers {
					if _, ok := containerMap[container.Name]; !ok {
						allErrs = append(allErrs, field.Invalid(containerOverridePath, override.Containers, fmt.Sprintf("container: %s, configured in the containerOverride should be present in the referenced training runtime", container.Name)))
					}
				}
			}
			if len(override.InitContainers) != 0 {
				initContainerMap := map[string]bool{}
				for _, job := range trainingRuntime.Spec.Template.Spec.ReplicatedJobs {
					for _, initContainer := range job.Template.Spec.Template.Spec.InitContainers {
						initContainerMap[initContainer.Name] = true
					}
				}
				initContainerOverridePath := podSpecOverridesPath.Index(idx)
				for _, container := range override.Containers {
					if _, ok := initContainerMap[container.Name]; !ok {
						allErrs = append(allErrs, field.Invalid(initContainerOverridePath, override.InitContainers, fmt.Sprintf("initContainer: %s, configured in the initContainerOverride should be present in the referenced training runtime", container.Name)))
					}
				}
			}
		}
	}
	return allErrs
}
