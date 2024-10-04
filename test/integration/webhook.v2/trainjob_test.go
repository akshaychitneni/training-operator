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
	kubeflowv2 "github.com/kubeflow/training-operator/pkg/apis/kubeflow.org/v2alpha1"
	"github.com/kubeflow/training-operator/test/integration/framework"
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	jobsetv1alpha2 "sigs.k8s.io/jobset/api/jobset/v1alpha2"
)

var _ = ginkgo.Describe("TrainJob Webhook", ginkgo.Ordered, func() {
	var ns *corev1.Namespace
	apiGroup := "kubeflow.org"
	runtimeKind := "TrainingRuntime"

	ginkgo.BeforeAll(func() {
		fwk = &framework.Framework{}
		cfg = fwk.Init()
		ctx, k8sClient = fwk.RunManager(cfg)
	})
	ginkgo.AfterAll(func() {
		fwk.Teardown()
	})

	ginkgo.BeforeEach(func() {
		ns = &corev1.Namespace{
			TypeMeta: metav1.TypeMeta{
				APIVersion: corev1.SchemeGroupVersion.String(),
				Kind:       "Namespace",
			},
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "trainjob-webhook-",
			},
		}
		gomega.Expect(k8sClient.Create(ctx, ns)).To(gomega.Succeed())
	})

	ginkgo.It("Should succeed in creating trainJob with trainingRuntime present", func() {

		runtimeName := "valid"
		trainingRuntime := kubeflowv2.TrainingRuntime{
			TypeMeta: metav1.TypeMeta{
				APIVersion: kubeflowv2.SchemeGroupVersion.String(),
				Kind:       runtimeKind,
			},
			ObjectMeta: metav1.ObjectMeta{
				Name:      runtimeName,
				Namespace: ns.Name,
			},
		}
		gomega.Expect(k8sClient.Create(ctx, &trainingRuntime)).To(gomega.Succeed())

		trainingRuntimeRef := kubeflowv2.TrainingRuntimeRef{
			Name:     runtimeName,
			APIGroup: &apiGroup,
			Kind:     &runtimeKind,
		}
		jobSpec := kubeflowv2.TrainJobSpec{
			TrainingRuntimeRef: trainingRuntimeRef,
		}
		trainJob := &kubeflowv2.TrainJob{
			TypeMeta: metav1.TypeMeta{
				APIVersion: kubeflowv2.SchemeGroupVersion.String(),
				Kind:       "TrainJob",
			},
			ObjectMeta: metav1.ObjectMeta{
				Name:      "valid-trainjob",
				Namespace: ns.Name,
			},
			Spec: jobSpec,
		}
		gomega.Expect(k8sClient.Create(ctx, trainJob)).To(gomega.Succeed())
	})

	ginkgo.It("Should fail in creating trainJob with trainingRuntime not present", func() {

		runtimeName := "inValid"

		trainingRuntimeRef := kubeflowv2.TrainingRuntimeRef{
			Name:     runtimeName,
			APIGroup: &apiGroup,
			Kind:     &runtimeKind,
		}
		jobSpec := kubeflowv2.TrainJobSpec{
			TrainingRuntimeRef: trainingRuntimeRef,
		}
		trainJob := &kubeflowv2.TrainJob{
			TypeMeta: metav1.TypeMeta{
				APIVersion: kubeflowv2.SchemeGroupVersion.String(),
				Kind:       "TrainJob",
			},
			ObjectMeta: metav1.ObjectMeta{
				Name:      "valid-trainjob",
				Namespace: ns.Name,
			},
			Spec: jobSpec,
		}
		gomega.Expect(k8sClient.Create(ctx, trainJob)).To(gomega.HaveOccurred())
	})

	ginkgo.It("Should fail in creating trainJob with pre-trained model config when referencing "+
		"a trainingRuntime without an initializer", func() {

		runtimeName := "runtime-without-initializer"
		trainingRuntime := kubeflowv2.TrainingRuntime{
			TypeMeta: metav1.TypeMeta{
				APIVersion: kubeflowv2.SchemeGroupVersion.String(),
				Kind:       runtimeKind,
			},
			ObjectMeta: metav1.ObjectMeta{
				Name:      runtimeName,
				Namespace: ns.Name,
			},
		}
		gomega.Expect(k8sClient.Create(ctx, &trainingRuntime)).To(gomega.Succeed())

		trainingRuntimeRef := kubeflowv2.TrainingRuntimeRef{
			Name:     runtimeName,
			APIGroup: &apiGroup,
			Kind:     &runtimeKind,
		}
		inputModel := &kubeflowv2.InputModel{}
		jobSpec := kubeflowv2.TrainJobSpec{
			TrainingRuntimeRef: trainingRuntimeRef,
			ModelConfig:        &kubeflowv2.ModelConfig{Input: inputModel},
		}
		trainJob := &kubeflowv2.TrainJob{
			TypeMeta: metav1.TypeMeta{
				APIVersion: kubeflowv2.SchemeGroupVersion.String(),
				Kind:       "TrainJob",
			},
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "invalid-trainjob-",
				Namespace:    ns.Name,
			},
			Spec: jobSpec,
		}
		gomega.Expect(k8sClient.Create(ctx, trainJob)).
			To(gomega.MatchError(gomega.ContainSubstring("admission webhook \"validator.trainjob.kubeflow.org\" " +
				"denied the request: spec.modelConfig.input")))
	})

	ginkgo.It("Should fail in creating trainJob with output model config when referencing a trainingRuntime"+
		" without an exporter", func() {

		runtimeName := "runtime-without-initializer"
		trainingRuntime := kubeflowv2.TrainingRuntime{
			TypeMeta: metav1.TypeMeta{
				APIVersion: kubeflowv2.SchemeGroupVersion.String(),
				Kind:       runtimeKind,
			},
			ObjectMeta: metav1.ObjectMeta{
				Name:      runtimeName,
				Namespace: ns.Name,
			},
		}
		gomega.Expect(k8sClient.Create(ctx, &trainingRuntime)).To(gomega.Succeed())

		trainingRuntimeRef := kubeflowv2.TrainingRuntimeRef{
			Name:     runtimeName,
			APIGroup: &apiGroup,
			Kind:     &runtimeKind,
		}
		outputModel := &kubeflowv2.OutputModel{}
		jobSpec := kubeflowv2.TrainJobSpec{
			TrainingRuntimeRef: trainingRuntimeRef,
			ModelConfig:        &kubeflowv2.ModelConfig{Output: outputModel},
		}
		trainJob := &kubeflowv2.TrainJob{
			TypeMeta: metav1.TypeMeta{
				APIVersion: kubeflowv2.SchemeGroupVersion.String(),
				Kind:       "TrainJob",
			},
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "invalid-trainjob-",
				Namespace:    ns.Name,
			},
			Spec: jobSpec,
		}
		gomega.Expect(k8sClient.Create(ctx, trainJob)).
			To(gomega.MatchError(gomega.ContainSubstring("admission webhook \"validator.trainjob.kubeflow.org\" " +
				"denied the request: spec.modelConfig.output")))
	})

	ginkgo.It("Should fail in creating trainJob with podSpecOverrides when referencing a trainingRuntime doesnt "+
		"have the job specified in the override", func() {

		runtimeName := "runtime-with-replicatedjob"
		trainingRuntimeSpec := kubeflowv2.TrainingRuntimeSpec{
			Template: kubeflowv2.JobSetTemplateSpec{
				Spec: jobsetv1alpha2.JobSetSpec{
					ReplicatedJobs: []jobsetv1alpha2.ReplicatedJob{
						{
							Name: "valid",
							Template: batchv1.JobTemplateSpec{
								Spec: batchv1.JobSpec{
									Template: corev1.PodTemplateSpec{
										Spec: corev1.PodSpec{
											Containers: []corev1.Container{
												{
													Name: "initializer",
												},
											},
										},
									},
								},
							},
						},
					},
				},
			},
		}
		trainingRuntime := kubeflowv2.TrainingRuntime{
			TypeMeta: metav1.TypeMeta{
				APIVersion: kubeflowv2.SchemeGroupVersion.String(),
				Kind:       runtimeKind,
			},
			ObjectMeta: metav1.ObjectMeta{
				Name:      runtimeName,
				Namespace: ns.Name,
			},
			Spec: trainingRuntimeSpec,
		}
		gomega.Expect(k8sClient.Create(ctx, &trainingRuntime)).To(gomega.Succeed())

		trainingRuntimeRef := kubeflowv2.TrainingRuntimeRef{
			Name:     runtimeName,
			APIGroup: &apiGroup,
			Kind:     &runtimeKind,
		}
		jobSpec := kubeflowv2.TrainJobSpec{
			TrainingRuntimeRef: trainingRuntimeRef,
			PodSpecOverrides: []kubeflowv2.PodSpecOverride{
				{TargetReplicatedJobs: []string{"valid", "invalid"}},
			},
		}
		trainJob := &kubeflowv2.TrainJob{
			TypeMeta: metav1.TypeMeta{
				APIVersion: kubeflowv2.SchemeGroupVersion.String(),
				Kind:       "TrainJob",
			},
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "invalid-trainjob-",
				Namespace:    ns.Name,
			},
			Spec: jobSpec,
		}
		gomega.Expect(k8sClient.Create(ctx, trainJob)).
			To(gomega.MatchError(gomega.ContainSubstring("admission webhook \"validator.trainjob.kubeflow.org\" " +
				"denied the request: spec.podSpecOverrides")))
	})
})
