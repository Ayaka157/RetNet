# Train a model using TPU v5e

> [!CAUTION]
> **Caution:** This page applies to the Cloud TPU API, which is no longer under active development. The Cloud TPU API will receive bug fixes and security updates only. For the latest features and support for the latest TPU versions, we recommend using Compute Engine or Google Kubernetes Engine to manage TPU resources. For more information, see [TPU resources in Compute Engine](https://docs.cloud.google.com/tpu/docs/tpus-in-compute-engine) and [About TPUs in GKE](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/tpus). The Cloud TPU API is not supported on TPU7x and later TPU versions.

With a smaller 256-chip footprint per Pod, TPU v5e is optimized to be a high-value
product for transformer, text-to-image, and Convolutional Neural Network (CNN)
training, fine-tuning, and serving. For more information about using
Cloud TPU v5e for serving, see [Inference using v5e](https://docs.cloud.google.com/tpu/docs/v5e-inference).

For more information about Cloud TPU v5e TPU hardware and configurations, see
[TPU v5e](https://docs.cloud.google.com/tpu/docs/v5e).

## Get started

The following sections describe how to get started using TPU v5e.

### Request quota

You need quota to use TPU v5e for training. There are different quota types for
on-demand TPUs, reserved TPUs, and TPU Spot VMs. There are separate
quotas required if you're using your TPU v5e for
[inference](https://docs.cloud.google.com/tpu/docs/v5e-inference). For more information about quotas, see
[Quotas](https://docs.cloud.google.com/tpu/docs/quota). To request TPU v5e quota, contact [Cloud
Sales](https://cloud.google.com/contact).

### Create a Google Cloud account and project

You need a Google Cloud account and project to use Cloud TPU. For more
information, see [Set up a Cloud TPU environment](https://docs.cloud.google.com/tpu/docs/setup-gcp-account).

### Create a Cloud TPU

The best practice is to provision Cloud TPU v5es as [queued resources](https://docs.cloud.google.com/tpu/docs/queued-resources)
using the `queued-resource create` command. For more information, see
[Manage queued resources](https://docs.cloud.google.com/tpu/docs/queued-resources).

You can also use the Create Node API (`gcloud compute tpus tpu-vm create`) to
provision Cloud TPU v5es. For more information, see [Manage TPU resources](https://docs.cloud.google.com/tpu/docs/managing-tpus-tpu-vm).

For more information about available v5e configurations for training, see
[Cloud TPU v5e types for training](https://docs.cloud.google.com/tpu/docs/v5e#tpu-v5e-config).

## Framework setup

This section describes the general setup process for custom model training using
JAX or PyTorch with TPU v5e.

For inference setup instructions, see [v5e inference introduction](https://docs.cloud.google.com/tpu/docs/v5e-inference).

Define some environment variables:

    export PROJECT_ID=your_project_ID
    export ACCELERATOR_TYPE=v5litepod-16
    export ZONE=us-west4-a
    export TPU_NAME=your_tpu_name
    export QUEUED_RESOURCE_ID=your_queued_resource_id

### Setup for JAX

If you have slice shapes greater than 8 chips, you will have multiple VMs in one
slice. In this case, you need to use the `--worker=all` flag to run the
installation on all TPU VMs in a single step without using SSH to log into each
separately:

    gcloud compute tpus tpu-vm ssh $TPU_NAME  \
       --project=$PROJECT_ID \
       --zone=$ZONE \
       --worker=all \
       --command='pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html'

#### Command flag descriptions

- `TPU_NAME`: The user-assigned text ID of the TPU which is created when the queued resource request is allocated.
- `PROJECT_ID`: Google Cloud Project Name. Use an existing project or create a new one at [Set up your
  Google Cloud project](https://docs.cloud.google.com/tpu/docs/setup-gcp-account#set-up-project)
- `ZONE`: See the [TPU regions and zones](https://docs.cloud.google.com/tpu/docs/regions-zones) document for the supported zones.
- `worker`: The TPU VM that has access to the underlying TPUs.

You can run the following command to check the number of devices (the outputs
shown here were produced with a v5litepod-16 slice). This code tests that
everything is installed correctly by checking that JAX sees the Cloud TPU
TensorCores and can run basic operations:

    gcloud compute tpus tpu-vm ssh $TPU_NAME \
       --project=$PROJECT_ID \
       --zone=$ZONE \
       --worker=all \
       --command='python3 -c "import jax; print(jax.device_count()); print(jax.local_device_count())"'

The output will be similar to the following:

    SSH: Attempting to connect to worker 0...
    SSH: Attempting to connect to worker 1...
    SSH: Attempting to connect to worker 2...
    SSH: Attempting to connect to worker 3...
    16
    4
    16
    4
    16
    4
    16
    4

`jax.device_count()` shows the total number of chips in the given slice.
`jax.local_device_count()` indicates the count of chips accessible by a single
VM in this slice.

    # Check the number of chips in the given slice by summing the count of chips
    # from all VMs through the
    # jax.local_device_count() API call.
    gcloud compute tpus tpu-vm ssh $TPU_NAME \
       --project=$PROJECT_ID \
       --zone=$ZONE \
       --worker=all \
       --command='python3 -c "import jax; xs=jax.numpy.ones(jax.local_device_count()); print(jax.pmap(lambda x: jax.lax.psum(x, \"i\"), axis_name=\"i\")(xs))"'

The output will be similar to the following:

    SSH: Attempting to connect to worker 0...
    SSH: Attempting to connect to worker 1...
    SSH: Attempting to connect to worker 2...
    SSH: Attempting to connect to worker 3...
    [16. 16. 16. 16.]
    [16. 16. 16. 16.]
    [16. 16. 16. 16.]
    [16. 16. 16. 16.]

Try the [JAX Tutorials](https://docs.cloud.google.com/tpu/docs/v5e-training#jax-flax-examples) in this document to get started with
v5e training using JAX.

### Setup for PyTorch

Note that v5e only supports the [PJRT runtime](https://pytorch.org/xla/master/runtime.html)
and PyTorch 2.1+ will use PJRT as the default runtime for all TPU versions.

This section describes how to start using PJRT on v5e with PyTorch/XLA with
commands for all workers.

#### Install dependencies

    gcloud compute tpus tpu-vm ssh $TPU_NAME  \
       --project=$PROJECT_ID \
       --zone=$ZONE \
       --worker=all \
       --command='
          sudo apt-get update -y
          sudo apt-get install libomp5 -y
          pip install mkl mkl-include
          pip install tf-nightly tb-nightly tbp-nightly
          pip install numpy
          sudo apt-get install libopenblas-dev -y
          pip install torch~=PYTORCH_VERSION torchvision torch_xla[tpu]~=PYTORCH_VERSION -f https://storage.googleapis.com/libtpu-releases/index.html -f https://storage.googleapis.com/libtpu-wheels/index.html'

Replace `PYTORCH_VERSION` with the version of PyTorch you want to use.
`PYTORCH_VERSION` is used to specify the same version for PyTorch/XLA. 2.6.0
is recommended.

For more information about versions of PyTorch and PyTorch/XLA, see
[PyTorch - Get Started](https://pytorch.org/get-started/locally/) and
[PyTorch/XLA releases](https://github.com/pytorch/xla/releases).

For more information on installing PyTorch/XLA, see [PyTorch/XLA installation](https://github.com/pytorch/xla?tab=readme-ov-file#tpu).

If you get an error when installing the wheels for `torch`, `torch_xla`, or
`torchvision` like
`pkg_resources.extern.packaging.requirements.InvalidRequirement: Expected end
or semicolon (after name and no valid version specifier) torch==nightly+20230222`,
downgrade your version with this command:

    pip3 install setuptools==62.1.0

#### Run a script with PJRT

> [!NOTE]
> **Note:** For models which have sizable, frequent allocations, using `tcmalloc` can significantly improve training time compared to the default `malloc` implementation. The default `malloc` used on a TPU VM is `tcmalloc`. However, depending on your workload (for example, with DLRM which has very large allocations for its embedding tables), `tcmalloc` may cause a slowdown. In this case, you might try to unset the following variable to use the default `malloc` instead:

    unset LD_PRELOAD

The following is an example using a Python script to do a calculation
on a v5e VM:

    gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
       --project=${PROJECT_ID} \
       --zone=${ZONE} \
       --worker=all \
       --command='
          export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.local/lib/
          export PJRT_DEVICE=TPU
          export PT_XLA_DEBUG=0
          export USE_TORCH=ON
          unset LD_PRELOAD
          export TPU_LIBRARY_PATH=$HOME/.local/lib/python3.10/site-packages/libtpu/libtpu.so
          python3 -c "import torch; import torch_xla; import torch_xla.core.xla_model as xm; print(xm.xla_device()); dev = xm.xla_device(); t1 = torch.randn(3,3,device=dev); t2 = torch.randn(3,3,device=dev); print(t1 + t2)"'

This generates output similar to the following:

    SSH: Attempting to connect to worker 0...
    SSH: Attempting to connect to worker 1...
    xla:0
    tensor([[ 1.8611, -0.3114, -2.4208],
    [-1.0731, 0.3422, 3.1445],
    [ 0.5743, 0.2379, 1.1105]], device='xla:0')
    xla:0
    tensor([[ 1.8611, -0.3114, -2.4208],
    [-1.0731, 0.3422, 3.1445],
    [ 0.5743, 0.2379, 1.1105]], device='xla:0')

Try the [PyTorch Tutorials](https://docs.cloud.google.com/tpu/docs/v5e-training#pytorch-xla) in this document to get started with
v5e training using PyTorch.

Delete your TPU and queued resource at the end of your session. To delete a
queued resource, delete the slice and then the queued resource in 2 steps:

    gcloud compute tpus tpu-vm delete $TPU_NAME \
       --project=$PROJECT_ID \
       --zone=$ZONE \
       --quiet

    gcloud compute tpus queued-resources delete $QUEUED_RESOURCE_ID \
       --project=$PROJECT_ID \
       --zone=${ZONE} \
       --quiet

These two steps can also be used to remove queued resource requests that are in
the `FAILED` state.

## JAX/FLAX examples

The following sections describe examples of how to train JAX and FLAX models on
TPU v5e.

### Train ImageNet on v5e

This tutorial describes how to train ImageNet on v5e using fake input data. If
you want to use real data, refer to the [README file on GitHub](https://github.com/google/flax/blob/main/examples/imagenet/README.md).

#### Set up

1. Create environment variables:


   ```bash
   export PROJECT_ID=your-project-id
   export TPU_NAME=your-tpu-name
   export ZONE=us-west4-a
   export ACCELERATOR_TYPE=v5litepod-8
   export RUNTIME_VERSION=v2-alpha-tpuv5-lite
   export SERVICE_ACCOUNT=your-service-account
   export QUEUED_RESOURCE_ID=your-queued-resource-id
   ```

   #### Environment variable descriptions

   - `PROJECT_ID`: Your Google Cloud project ID. Use an existing project or [create a new one.](https://docs.cloud.google.com/tpu/docs/setup-gcp-account#set-up-project)
   - `TPU_NAME`: The name of the TPU.
   - `ZONE`: The zone in which to create the TPU VM. For more information about supported zones, see [TPU regions and zones](https://docs.cloud.google.com/tpu/docs/regions-zones).
   - `ACCELERATOR_TYPE`: The accelerator type specifies the version and size of the Cloud TPU you want to create. For more information about supported accelerator types for each TPU version, see [TPU versions](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm#versions).
   - `RUNTIME_VERSION`: The Cloud TPU [software version](https://docs.cloud.google.com/tpu/docs/runtimes).
   - `SERVICE_ACCOUNT`: The email address for your service account. You can find it by going to the [Service Accounts page](https://console.cloud.google.com/iam-admin/serviceaccounts) in the Google Cloud console.


     For example:
     `tpu-service-account@PROJECT_ID.iam.gserviceaccount.com`
   - `QUEUED_RESOURCE_ID`: The user-assigned text ID of the queued resource request.

   <br />

2. [Create a TPU resource:](https://docs.cloud.google.com/tpu/docs/queued-resources)

       gcloud compute tpus queued-resources create $QUEUED_RESOURCE_ID \
          --node-id=$TPU_NAME \
          --project=$PROJECT_ID \
          --zone=$ZONE \
          --accelerator-type=$ACCELERATOR_TYPE \
          --runtime-version=$RUNTIME_VERSION \
          --service-account=$SERVICE_ACCOUNT

   > [!NOTE]
   > **Note:** To use a reservation, add the `--reserved` flag. To use TPU Spot VMs, add the `--spot` flag.

   You will be able to SSH to your TPU VM once your queued resource is in
   the `ACTIVE` state:

       gcloud compute tpus queued-resources describe $QUEUED_RESOURCE_ID  \
          --project=$PROJECT_ID \
          --zone=$ZONE

   When the QueuedResource is in the `ACTIVE` state, the output will
   be similar to the following:

        state: ACTIVE

3. Install newest version of JAX and jaxlib:

       gcloud compute tpus tpu-vm ssh $TPU_NAME \
          --project=$PROJECT_ID \
          --zone=$ZONE \
          --worker=all \
          --command='pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html'

   > [!NOTE]
   > **Note:** If you see an error suggesting you use jax\>=0.4.8, you can safely ignore the message.

4. Clone the ImageNet model and install the corresponding requirements:

       gcloud compute tpus tpu-vm ssh $TPU_NAME \
          --project=$PROJECT_ID \
          --zone=$ZONE \
          --worker=all \
          --command="git clone https://github.com/coolkp/flax.git && cd flax && git checkout pmap-orbax-conversion && git pull"

5. To generate fake data, the model needs information on the dimensions of
   the dataset. This can be gathered from the ImageNet dataset's metadata:

       gcloud compute tpus tpu-vm ssh $TPU_NAME \
          --project=$PROJECT_ID \
          --zone=$ZONE \
          --worker=all \
          --command="cd flax/examples/imagenet && pip install -r requirements-cloud-tpu.txt"

#### Train the model

Once all the previous steps are done, you can train the model.

    gcloud compute tpus tpu-vm ssh $TPU_NAME \
       --project=$PROJECT_ID \
       --zone=$ZONE \
       --worker=all \
       --command="cd flax/examples/imagenet && bash ../../tests/download_dataset_metadata.sh && JAX_PLATFORMS=tpu python imagenet_fake_data_benchmark.py"

#### Delete the TPU and queued resource

Delete your TPU and queued resource at the end of your session.

    gcloud compute tpus tpu-vm delete $TPU_NAME \
       --project=$PROJECT_ID \
       --zone=$ZONE \
       --quiet

    gcloud compute tpus queued-resources delete $QUEUED_RESOURCE_ID \
       --project=$PROJECT_ID \
       --zone=$ZONE \
       --quiet

### Hugging Face FLAX Models

[Hugging Face](https://huggingface.co/) models implemented in FLAX work out of
the box on Cloud TPU v5e. This section provides instructions for running
popular models.

#### Train ViT on Imagenette

This tutorial shows you how to train the [Vision Transformer](https://arxiv.org/abs/2010.11929)
(ViT) model from HuggingFace using the Fast AI [Imagenette](https://github.com/fastai/imagenette)
dataset on Cloud TPU v5e.

The ViT model was the first one that successfully trained a Transformer encoder
on ImageNet with excellent results compared to convolutional networks. For more
information, see [ViT overview](https://huggingface.co/docs/transformers/model_doc/vit).

#### Set up

1. Create environment variables:


   ```bash
   export PROJECT_ID=your-project-id
   export TPU_NAME=your-tpu-name
   export ZONE=us-west4-a
   export ACCELERATOR_TYPE=v5litepod-16
   export RUNTIME_VERSION=v2-alpha-tpuv5-lite
   export SERVICE_ACCOUNT=your-service-account
   export QUEUED_RESOURCE_ID=your-queued-resource-id
   ```

   #### Environment variable descriptions

   - `PROJECT_ID`: Your Google Cloud project ID. Use an existing project or [create a new one.](https://docs.cloud.google.com/tpu/docs/setup-gcp-account#set-up-project)
   - `TPU_NAME`: The name of the TPU.
   - `ZONE`: The zone in which to create the TPU VM. For more information about supported zones, see [TPU regions and zones](https://docs.cloud.google.com/tpu/docs/regions-zones).
   - `ACCELERATOR_TYPE`: The accelerator type specifies the version and size of the Cloud TPU you want to create. For more information about supported accelerator types for each TPU version, see [TPU versions](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm#versions).
   - `RUNTIME_VERSION`: The Cloud TPU [software version](https://docs.cloud.google.com/tpu/docs/runtimes).
   - `SERVICE_ACCOUNT`: The email address for your service account. You can find it by going to the [Service Accounts page](https://console.cloud.google.com/iam-admin/serviceaccounts) in the Google Cloud console.


     For example:
     `tpu-service-account@PROJECT_ID.iam.gserviceaccount.com`
   - `QUEUED_RESOURCE_ID`: The user-assigned text ID of the queued resource request.

   <br />

2. [Create a TPU resource:](https://docs.cloud.google.com/tpu/docs/queued-resources)

       gcloud compute tpus queued-resources create $QUEUED_RESOURCE_ID \
          --node-id=$TPU_NAME \
          --project=$PROJECT_ID \
          --zone=$ZONE \
          --accelerator-type=$ACCELERATOR_TYPE \
          --runtime-version=$RUNTIME_VERSION \
          --service-account=$SERVICE_ACCOUNT

   > [!NOTE]
   > **Note:** To use a reservation, add the `--reserved` flag. To use TPU Spot VMs, add the `--spot` flag.

   You will be able to SSH to your TPU VM once your queued resource
   is in state `ACTIVE`:

       gcloud compute tpus queued-resources describe $QUEUED_RESOURCE_ID  \
          --project=$PROJECT_ID \
          --zone=$ZONE

   When the queued resource is in the `ACTIVE` state, the output will be similar
   to the following:

        state: ACTIVE

3. Install JAX and its library:

       gcloud compute tpus tpu-vm ssh $TPU_NAME \
          --project=$PROJECT_ID \
          --zone=$ZONE \
          --worker=all \
          --command='pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html'

4. Download Hugging Face [repository](https://github.com/huggingface/transformers.git)
   and install requirements:

       gcloud compute tpus tpu-vm ssh $TPU_NAME \
          --project=$PROJECT_ID \
          --zone=$ZONE \
          --worker=all \
          --command='git clone https://github.com/huggingface/transformers.git && cd transformers && pip install . && pip install -r examples/flax/_tests_requirements.txt && pip install --upgrade huggingface-hub urllib3 zipp && pip install tensorflow==2.19 && sed -i "s/torchvision==0.12.0+cpu/torchvision==0.22.1/" examples/flax/vision/requirements.txt && pip install -r examples/flax/vision/requirements.txt && pip install tf-keras'

5. Download the Imagenette dataset:

       gcloud compute tpus tpu-vm ssh $TPU_NAME \
          --project=$PROJECT_ID \
          --zone=$ZONE \
          --worker=all \
          --command='cd transformers && wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz && tar -xvzf imagenette2.tgz'

#### Train the model

Train the model with a pre-mapped buffer at 4GB.

    gcloud compute tpus tpu-vm ssh $TPU_NAME \
       --project=$PROJECT_ID \
       --zone=$ZONE \
       --worker=all \
       --command='cd transformers && JAX_PLATFORMS=tpu python3 examples/flax/vision/run_image_classification.py --train_dir "imagenette2/train" --validation_dir "imagenette2/val" --output_dir "./vit-imagenette" --learning_rate 1e-3 --preprocessing_num_workers 32 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --model_name_or_path google/vit-base-patch16-224-in21k --num_train_epochs 3'

#### Delete the TPU and queued resource

Delete your TPU and queued-resource at the end of your session.

    gcloud compute tpus tpu-vm delete $TPU_NAME \
       --project=$PROJECT_ID \
       --zone=$ZONE \
       --quiet

    gcloud compute tpus queued-resources delete $QUEUED_RESOURCE_ID \
       --project=$PROJECT_ID \
       --zone=$ZONE \
       --quiet

#### ViT benchmarking results

The training script was run on v5litepod-4, v5litepod-16, and v5litepod-64. The
following table shows the throughputs with different accelerator types.

|---|---|---|---|
| Accelerator type | v5litepod-4 | v5litepod-16 | v5litepod-64 |
| Epoch | 3 | 3 | 3 |
| Global batch size | 32 | 128 | 512 |
| Throughput (examples/sec) | 263.40 | 429.34 | 470.71 |

#### Train Diffusion on Pokémon

This tutorial shows you how to train the Stable Diffusion model from
HuggingFace using the [Pokémon](https://huggingface.co/datasets/diffusers/pokemon-gpt4-captions)
dataset on Cloud TPU v5e.

The Stable Diffusion model is a latent text-to-image model that generates
photorealistic images from any text input. For more information, see the
following resources:

- [Stable Diffusion overview](https://huggingface.co/CompVis/stable-diffusion-v1-4)
- [Stable Diffusion code source](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image)

#### Set up

1. Set an environment variable for the name of your storage bucket:

       export GCS_BUCKET_NAME=your_bucket_name

2. Set up a storage bucket for your model output:

       Google Cloud CLI storage buckets create gs://$GCS_BUCKET_NAME \
            --project=$PROJECT_ID \
            --location=$ZONE

3. Create environment variables:


   ```bash
   export PROJECT_ID=your-project-id
   export TPU_NAME=your-tpu-name
   export ZONE=us-west1-c
   export ACCELERATOR_TYPE=v5litepod-16
   export RUNTIME_VERSION=v2-alpha-tpuv5-lite
   export SERVICE_ACCOUNT=your-service-account
   export QUEUED_RESOURCE_ID=your-queued-resource-id
   ```

   #### Environment variable descriptions

   - `PROJECT_ID`: Your Google Cloud project ID. Use an existing project or [create a new one.](https://docs.cloud.google.com/tpu/docs/setup-gcp-account#set-up-project)
   - `TPU_NAME`: The name of the TPU.
   - `ZONE`: The zone in which to create the TPU VM. For more information about supported zones, see [TPU regions and zones](https://docs.cloud.google.com/tpu/docs/regions-zones).
   - `ACCELERATOR_TYPE`: The accelerator type specifies the version and size of the Cloud TPU you want to create. For more information about supported accelerator types for each TPU version, see [TPU versions](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm#versions).
   - `RUNTIME_VERSION`: The Cloud TPU [software version](https://docs.cloud.google.com/tpu/docs/runtimes).
   - `SERVICE_ACCOUNT`: The email address for your service account. You can find it by going to the [Service Accounts page](https://console.cloud.google.com/iam-admin/serviceaccounts) in the Google Cloud console.


     For example:
     `tpu-service-account@PROJECT_ID.iam.gserviceaccount.com`
   - `QUEUED_RESOURCE_ID`: The user-assigned text ID of the queued resource request.

   <br />

4. [Create a TPU resource:](https://docs.cloud.google.com/tpu/docs/queued-resources)

       gcloud compute tpus queued-resources create $QUEUED_RESOURCE_ID \
          --node-id=$TPU_NAME \
          --project=$PROJECT_ID \
          --zone=$ZONE \
          --accelerator-type=$ACCELERATOR_TYPE \
          --runtime-version=$RUNTIME_VERSION \
          --service-account=$SERVICE_ACCOUNT

   > [!NOTE]
   > **Note:** To use a reservation, add the `--reserved` flag. To use TPU Spot VMs, add the `--spot` flag.

   You will be able to SSH to your TPU VM once your queued resource is in the `ACTIVE` state:

       gcloud compute tpus queued-resources describe $QUEUED_RESOURCE_ID  \
          --project=$PROJECT_ID \
          --zone=$ZONE

   When the queued resource is in the `ACTIVE` state, the output will be similar to the following:

        state: ACTIVE

5. Install JAX and its library.

       gcloud compute tpus tpu-vm ssh $TPU_NAME \
          --project=$PROJECT_ID \
          --zone=$ZONE \
          --worker=all \
          --command='pip install "jax[tpu]==0.4.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html'

6. Download the HuggingFace [repository](https://github.com/huggingface/diffusers)
   and install requirements.

       gcloud compute tpus tpu-vm ssh $TPU_NAME \
            --project=$PROJECT_ID \
            --zone=$ZONE \
            --worker=all \
            --command='git clone https://github.com/RissyRan/diffusers.git && cd diffusers && pip install . && pip install -U -r examples/text_to_image/requirements_flax.txt && pip install tensorflow==2.17.1 clu && pip install tensorboard==2.17.1'

#### Train the model

Train the model with a pre-mapped buffer at 4GB.

    gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT_ID --worker=all --command="
        git clone https://github.com/google/maxdiffusion
        cd maxdiffusion
        pip3 install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
        pip3 install -r requirements.txt
        pip3 install .
        pip3 install gcsfs
        export LIBTPU_INIT_ARGS=''
        python -m src.maxdiffusion.train src/maxdiffusion/configs/base_2_base.yml run_name=my_run \
        jax_cache_dir=gs://${GCS_BUCKET_NAME} activations_dtype=bfloat16 weights_dtype=bfloat16 \
        per_device_batch_size=1 precision=DEFAULT dataset_save_location=gs://${GCS_BUCKET_NAME} \
        output_dir=gs://${GCS_BUCKET_NAME}/ attention=flash"

#### Clean up

Delete your TPU, queued resource, and Cloud Storage bucket at the end of your
session.

1. Delete your TPU:

       gcloud compute tpus tpu-vm delete $TPU_NAME \
           --project=$PROJECT_ID \
           --zone=$ZONE \
           --quiet

2. Delete the queued resource:

       gcloud compute tpus queued-resources delete $QUEUED_RESOURCE_ID \
           --project=$PROJECT_ID \
           --zone=$ZONE \
           --quiet

3. Delete the Cloud Storage bucket:

       gcloud storage rm -r gs://$GCS_BUCKET_NAME

#### Benchmarking results for diffusion

The training script ran on v5litepod-4, v5litepod-16, and v5litepod-64.
The following table shows the throughputs.

|---|---|---|---|
| Accelerator type | v5litepod-4 | v5litepod-16 | v5litepod-64 |
| Train Step | 1500 | 1500 | 1500 |
| Global batch size | 32 | 64 | 128 |
| Throughput (examples/sec) | 36.53 | 43.71 | 49.36 |

## PyTorch/XLA

The following sections describe examples of how to train PyTorch/XLA models on
TPU v5e.

### Train ResNet using the PJRT runtime

PyTorch/XLA is migrating from XRT to PjRt from PyTorch 2.0+. Here are the
updated instructions to set up v5e for PyTorch/XLA training workloads.

##### Set up

1. Create environment variables:


   ```bash
   export PROJECT_ID=your-project-id
   export TPU_NAME=your-tpu-name
   export ZONE=us-west4-a
   export ACCELERATOR_TYPE=v5litepod-16
   export RUNTIME_VERSION=v2-alpha-tpuv5-lite
   export SERVICE_ACCOUNT=your-service-account
   export QUEUED_RESOURCE_ID=your-queued-resource-id
   ```

   #### Environment variable descriptions

   - `PROJECT_ID`: Your Google Cloud project ID. Use an existing project or [create a new one.](https://docs.cloud.google.com/tpu/docs/setup-gcp-account#set-up-project)
   - `TPU_NAME`: The name of the TPU.
   - `ZONE`: The zone in which to create the TPU VM. For more information about supported zones, see [TPU regions and zones](https://docs.cloud.google.com/tpu/docs/regions-zones).
   - `ACCELERATOR_TYPE`: The accelerator type specifies the version and size of the Cloud TPU you want to create. For more information about supported accelerator types for each TPU version, see [TPU versions](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm#versions).
   - `RUNTIME_VERSION`: The Cloud TPU [software version](https://docs.cloud.google.com/tpu/docs/runtimes).
   - `SERVICE_ACCOUNT`: The email address for your service account. You can find it by going to the [Service Accounts page](https://console.cloud.google.com/iam-admin/serviceaccounts) in the Google Cloud console.


     For example:
     `tpu-service-account@PROJECT_ID.iam.gserviceaccount.com`
   - `QUEUED_RESOURCE_ID`: The user-assigned text ID of the queued resource request.

   <br />

2. [Create a TPU resource:](https://docs.cloud.google.com/tpu/docs/queued-resources)

       gcloud compute tpus queued-resources create $QUEUED_RESOURCE_ID \
          --node-id=$TPU_NAME \
          --project=$PROJECT_ID \
          --zone=$ZONE \
          --accelerator-type=$ACCELERATOR_TYPE \
          --runtime-version=$RUNTIME_VERSION \
          --service-account=$SERVICE_ACCOUNT

   > [!NOTE]
   > **Note:** To use a reservation, add the `--reserved` flag. To use TPU Spot VMs, add the `--spot` flag.

   You will be able to SSH to your TPU VM once your QueuedResource
   is in `ACTIVE` state:

       gcloud compute tpus queued-resources describe $QUEUED_RESOURCE_ID \
          --project=$PROJECT_ID \
          --zone=$ZONE

   When the queued resource is in the `ACTIVE` state, the output will be similar
   to the following:

        state: ACTIVE

3. Install Torch/XLA specific dependencies

       gcloud compute tpus tpu-vm ssh $TPU_NAME \
          --project=$PROJECT_ID \
          --zone=$ZONE \
          --worker=all \
          --command='
             sudo apt-get update -y
             sudo apt-get install libomp5 -y
             pip3 install mkl mkl-include
             pip3 install tf-nightly tb-nightly tbp-nightly
             pip3 install numpy
             sudo apt-get install libopenblas-dev -y
             pip install torch==PYTORCH_VERSION torchvision torch_xla[tpu]==PYTORCH_VERSION -f https://storage.googleapis.com/libtpu-releases/index.html -f https://storage.googleapis.com/libtpu-wheels/index.html'

   Replace `PYTORCH_VERSION` with the version of PyTorch you want to use.
   `PYTORCH_VERSION` is used to specify the same version for PyTorch/XLA. 2.6.0
   is recommended.

   For more information about versions of PyTorch and PyTorch/XLA, see
   [PyTorch - Get Started](https://pytorch.org/get-started/locally/) and
   [PyTorch/XLA releases](https://github.com/pytorch/xla/releases).

   For more information on installing PyTorch/XLA, see [PyTorch/XLA installation](https://github.com/pytorch/xla?tab=readme-ov-file#tpu).

##### Train the ResNet model

    gcloud compute tpus tpu-vm ssh $TPU_NAME \
       --project=$PROJECT_ID \
       --zone=$ZONE \
       --worker=all \
       --command='
          date
          export PJRT_DEVICE=TPU
          export PT_XLA_DEBUG=0
          export USE_TORCH=ON
          export XLA_USE_BF16=1
          export LIBTPU_INIT_ARGS=--xla_jf_auto_cross_replica_sharding
          export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
          export TPU_LIBRARY_PATH=$HOME/.local/lib/python3.10/site-packages/libtpu/libtpu.so
          git clone https://github.com/pytorch/xla.git
          cd xla/
          git checkout release-r2.6
          python3 test/test_train_mp_imagenet.py --model=resnet50  --fake_data --num_epochs=1 --num_workers=16  --log_steps=300 --batch_size=64 --profile'

#### Delete the TPU and queued resource

Delete your TPU and queued resource at the end of your session.

    gcloud compute tpus tpu-vm delete $TPU_NAME \
       --project=$PROJECT_ID \
       --zone=$ZONE \
       --quiet

    gcloud compute tpus queued-resources delete $QUEUED_RESOURCE_ID \
       --project=$PROJECT_ID \
       --zone=$ZONE \
       --quiet

##### Benchmark result

The following table shows the benchmark throughputs.

|---|---|
| Accelerator type | Throughput (examples/second) |
| v5litepod-4 | 4240 ex/s |
| v5litepod-16 | 10,810 ex/s |
| v5litepod-64 | 46,154 ex/s |

### Train ViT on v5e

This tutorial will cover how to run VIT on v5e using the HuggingFace [repository](https://github.com/huggingface/transformers.git)
on PyTorch/XLA on the [cifar10 dataset](https://huggingface.co/datasets/cifar10).

#### Set up

1. Create environment variables:


   ```bash
   export PROJECT_ID=your-project-id
   export TPU_NAME=your-tpu-name
   export ZONE=us-west4-a
   export ACCELERATOR_TYPE=v5litepod-16
   export RUNTIME_VERSION=v2-alpha-tpuv5-lite
   export SERVICE_ACCOUNT=your-service-account
   export QUEUED_RESOURCE_ID=your-queued-resource-id
   ```

   #### Environment variable descriptions

   - `PROJECT_ID`: Your Google Cloud project ID. Use an existing project or [create a new one.](https://docs.cloud.google.com/tpu/docs/setup-gcp-account#set-up-project)
   - `TPU_NAME`: The name of the TPU.
   - `ZONE`: The zone in which to create the TPU VM. For more information about supported zones, see [TPU regions and zones](https://docs.cloud.google.com/tpu/docs/regions-zones).
   - `ACCELERATOR_TYPE`: The accelerator type specifies the version and size of the Cloud TPU you want to create. For more information about supported accelerator types for each TPU version, see [TPU versions](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm#versions).
   - `RUNTIME_VERSION`: The Cloud TPU [software version](https://docs.cloud.google.com/tpu/docs/runtimes).
   - `SERVICE_ACCOUNT`: The email address for your service account. You can find it by going to the [Service Accounts page](https://console.cloud.google.com/iam-admin/serviceaccounts) in the Google Cloud console.


     For example:
     `tpu-service-account@PROJECT_ID.iam.gserviceaccount.com`
   - `QUEUED_RESOURCE_ID`: The user-assigned text ID of the queued resource request.

   <br />

2. [Create a TPU resource:](https://docs.cloud.google.com/tpu/docs/queued-resources)

       gcloud compute tpus queued-resources create $QUEUED_RESOURCE_ID \
          --node-id=$TPU_NAME \
          --project=$PROJECT_ID \
          --zone=$ZONE \
          --accelerator-type=$ACCELERATOR_TYPE \
          --runtime-version=$RUNTIME_VERSION \
          --service-account=$SERVICE_ACCOUNT

   > [!NOTE]
   > **Note:** To use a reservation, add the `--reserved` flag. To use TPU Spot VMs, add the `--spot` flag.

   You will be able to SSH to your TPU VM once your QueuedResource is in
   the `ACTIVE` state:

        gcloud compute tpus queued-resources describe $QUEUED_RESOURCE_ID \
          --project=$PROJECT_ID \
          --zone=$ZONE

   When the queued resource is in the `ACTIVE` state, the output will
   be similar to the following:

        state: ACTIVE

3. Install PyTorch/XLA dependencies

       gcloud compute tpus tpu-vm ssh $TPU_NAME \
       --project=$PROJECT_ID \
       --zone=$ZONE \
       --worker=all \
       --command='
          sudo apt-get update -y
          sudo apt-get install libomp5 -y
          pip3 install mkl mkl-include
          pip3 install tf-nightly tb-nightly tbp-nightly
          pip3 install numpy
          sudo apt-get install libopenblas-dev -y
          pip install torch==PYTORCH_VERSION torchvision torch_xla[tpu]==PYTORCH_VERSION -f https://storage.googleapis.com/libtpu-releases/index.html -f https://storage.googleapis.com/libtpu-wheels/index.html
          pip install jax==0.4.38 jaxlib==0.4.38 -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/

   Replace `PYTORCH_VERSION` with the version of PyTorch you want to use.
   `PYTORCH_VERSION` is used to specify the same version for PyTorch/XLA. 2.6.0
   is recommended.

   For more information about versions of PyTorch and PyTorch/XLA, see
   [PyTorch - Get Started](https://pytorch.org/get-started/locally/) and
   [PyTorch/XLA releases](https://github.com/pytorch/xla/releases).

   For more information on installing PyTorch/XLA, see [PyTorch/XLA installation](https://github.com/pytorch/xla?tab=readme-ov-file#tpu).
4. Download HuggingFace [repository](https://github.com/huggingface/transformers.git)
   and install requirements.

          gcloud compute tpus tpu-vm ssh $TPU_NAME \
          --project=$PROJECT_ID \
          --zone=$ZONE \
          --worker=all \
          --command="
             git clone https://github.com/suexu1025/transformers.git vittransformers; \
             cd vittransformers; \
             pip3 install .; \
             pip3 install datasets; \
             wget https://github.com/pytorch/xla/blob/master/scripts/capture_profile.py"

#### Train the model

    gcloud compute tpus tpu-vm ssh $TPU_NAME \
       --project=$PROJECT_ID \
       --zone=$ZONE \
       --worker=all \
       --command='
          export PJRT_DEVICE=TPU
          export PT_XLA_DEBUG=0
          export USE_TORCH=ON
          export TF_CPP_MIN_LOG_LEVEL=0
          export XLA_USE_BF16=1
          export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
          export TPU_LIBRARY_PATH=$HOME/.local/lib/python3.10/site-packages/libtpu/libtpu.so
          cd vittransformers
          python3 -u examples/pytorch/xla_spawn.py --num_cores 4 examples/pytorch/image-pretraining/run_mae.py --dataset_name=cifar10 \
          --remove_unused_columns=False \
          --label_names=pixel_values \
          --mask_ratio=0.75 \
          --norm_pix_loss=True \
          --do_train=true \
          --do_eval=true \
          --base_learning_rate=1.5e-4 \
          --lr_scheduler_type=cosine \
          --weight_decay=0.05 \
          --num_train_epochs=3 \
          --warmup_ratio=0.05 \
          --per_device_train_batch_size=8 \
          --per_device_eval_batch_size=8 \
          --logging_strategy=steps \
          --logging_steps=30 \
          --evaluation_strategy=epoch \
          --save_strategy=epoch \
          --load_best_model_at_end=True \
          --save_total_limit=3 \
          --seed=1337 \
          --output_dir=MAE \
          --overwrite_output_dir=true \
          --logging_dir=./tensorboard-metrics \
          --tpu_metrics_debug=true'

#### Delete the TPU and queued resource

Delete your TPU and queued resource at the end of your session.

    gcloud compute tpus tpu-vm delete $TPU_NAME \
       --project=$PROJECT_ID \
       --zone=$ZONE \
       --quiet

    gcloud compute tpus queued-resources delete $QUEUED_RESOURCE_ID \
       --project=$PROJECT_ID \
       --zone=$ZONE \
       --quiet

#### Benchmark result

The following table shows the benchmark throughputs for different accelerator types.

|---|---|---|---|
|   | v5litepod-4 | v5litepod-16 | v5litepod-64 |
| Epoch | 3 | 3 | 3 |
| Global batch size | 32 | 128 | 512 |
| Throughput (examples/sec) | 201 | 657 | 2,844 |