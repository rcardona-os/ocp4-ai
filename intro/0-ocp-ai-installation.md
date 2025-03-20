# General Information about Installation

Depending on the specific features and components required from Red Hat OpenShift Data Science, the following operators configure the platform with comprehensive capabilities for creating, overseeing, and deploying the entire life cycle of a Machine Learning application.

[ 1 - Red Hat OpenShift Pipelines Operator](https://www.redhat.com/en/technologies/cloud-computing/openshift/pipelines): The Red Hat OpenShift Pipelines Operator is required if you want to install the Red Hat OpenShift Data Science Pipelines component.

[ 2 - Node Feature Discovery Operator](https://docs.openshift.com/container-platform/4.13/hardware_enablement/psap-node-feature-discovery-operator.html): The Node Feature Discovery Operator is a prerequisite for the NVIDIA GPU Operator.

[ 3 - NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html): The NVIDIA GPU Operator is required for GPU support in Red Hat OpenShift Data Science.

NOTE: It is possible to access Red Hat OpenShift Data Science for installation in two ways: as a self-managed version via OperatorHub or as a fully managed solution through the OpenShift Marketplace.

### Prerequisites
- OCPv14.3 or newer

- User with cluster admin privileges 

### Operators Installation

---

#### 1 - Data Science Pipelines Operator

  - Login to Red Hat OpenShift using a user which has the cluster-admin role assigned.

  - Navigate to __Operators → OperatorHub__ and search for __Red Hat OpenShift Pipelines__

    ![Pipeline-search](media/pipeline_search.png)
  
  - Click on the __Red Hat OpenShift Pipelines__ operator and in the pop up window click on __Install__ to open the operator’s installation view.

    ![pipeline_install1](media/pipeline_install1.png)

  - In the installation view choose the __Update channel__ and the __Update approval__ parameters. You can accept the default values. The __Installation mode__ and the __Installed namespace__ parameters are fixed.

    ![pipeline_install2](media/pipeline_install2.png)

  - Click on the Install button at the bottom of to view the to proceed with the installation. A window showing the installation progress will pop up.

    ![pipeline_install3](media/pipeline_install3.png)
  
  - When the installation finishes the operator is ready to be used by __Red Hat OpenShift Data Science__.

    ![pipeline_install4](media/pipeline_install4.png)

  - Red Hat OpenShift Pipelines is now successfully installed.

#### 2 - Feature Discovery Operator

  - Navigate to __Operators → OperatorHub__ and search for __Node Feature Discovery__

    ![nfd_search.png](media/nfd_search.png)

  - Two options for the __Node Feature Discovery__ operator will be available. Click on the one with Red Hat in the top right hand corner and in the pop up window click on Install to open the operator’s installation view. Make sure you select __Node Feature Discovery__ from Red Hat not the Community version.

    ![nfd_install1.png](media/nfd_install1.png)

  - In the installation view check the box to __Enable Operator recommended cluster monitoring on this Namespace__ and the __Update approval__ parameters if desired. Leave the __Update channel__, Version, and the Installed Namespace parameters as the default options. Some of these options may vary slightly depending on your version of OpenShift. Please refer to the official Node Feature Discovery Documentation for your version of OpenShift for the recommended settings.

    ![nfd_install2.png](media/nfd_install2.png)

  - Click on the Install button at the bottom of to view the to proceed with the installation. A window showing the installation progress will pop up.

    ![nfd_install3.png](media/nfd_install3.png)

  - When the installation finishes the operator to be configured. Click the button to __View Operator__.

    ![nfd_install4.png](media/nfd_install4.png)

  - Click the __Create instance__ button for the __NodeFeatureDiscovery__ object.

    ![nfd_configure1.png](media/nfd_configure1.png)

  - Leave the default options for __NodeFeatureDiscovery__ selected, and click the __Create__ button.

    ![nfd_configure2.png](media/nfd_configure2.png)

  - A new set of pods should appear in the __Workloads → Pods__ section managed by the nfd-worker DaemonSet. Node Feature Discovery will now be able to automatically detect information about the nodes in the cluster and apply labels to those nodes.

    ![nfd_verify.png](media/nfd_verify.png)

  - __Node Feature Discovery__ is now successfully installed and configured.

#### 3 - NVIDIA GPU Operator

  - Navigate to __Operators → OperatorHub__ and search for __NVIDIA GPU Operator__
 
    ![gpu_search](media/gpu_search.png)

  - Click the NVIDIA GPU Operator tile and in the pop up window click on Install to open the operator’s installation view.
 
    ![gpu_install1](media/gpu_install1.png)

  - In the installation view update the __Update channel__ and __Update approval__ parameters if desired. Leave the __Installation mode__ and the __Installed namespace__ parameters as the default options.
 
    ![gpu_install2](media/gpu_install2.png)

  - Click on the Install button at the bottom of to view the to proceed with the installation. A window showing the installation progress will pop up.
 
    ![gpu_install3](media/gpu_install3.png)

  - When the installation finishes the operator to be configured. Click the button to View Operator.
 
    ![gpu_install4](media/gpu_install4.png)

  - Click the __Create instance__ button for the __ClusterPolicy__ object.
 
    ![gpu_configure1](media/gpu_configure1.png)

  - Leave the default options for __ClusterPolicy__ selected, and click the __Create__ button.
 
    ![gpu_configure2](media/gpu_configure2.png)

  - After the gpu-cluster-policy __ClusterPolicy__ is created, the __NVIDIA GPU Operator__ will update the status of the __ClusterPolicy__ to __State: ready__.
 
    ![gpu_verify1](media/gpu_verify1.png)

  - After the Red Hat OpenShift Data Science operator has been installed and configured, users will be able to see an option for "Number of GPUs" when creating a new workbench.

    ![gpu_verify2](media/gpu_verify2.png)

  - The Dashboard may initially show "All GPUs are currently in use, try again later." when Red Hat OpenShift Data Science is first installed. It may take a few minutes after Red Hat OpenShift Data Science is installed before the GPUs are initially detected. The NVIDIA GPU Operator supports many advanced use cases such as Multi-Instance GPU (MIG) and Time Slicing that are configurable using the ClusterPolicy. For information about advanced GPU configuration capabilities, refer to the official [NVIDIA Documentation](https://docs.nvidia.com/datacenter/cloud-native/openshift/latest/introduction.html).