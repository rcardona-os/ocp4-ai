$ cat << EOF | oc apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: pvc-inspect
spec:
  containers:
  - name: shell
    image: registry.access.redhat.com/ubi8/ubi
    command: ["/bin/sh", "-c", "sleep 3600"]
    volumeMounts:
    - name: model-storage
      mountPath: /workspace
  volumes:
  - name: model-storage
    persistentVolumeClaim:
      claimName: model-storage
  restartPolicy: Never
EOF


$ cat << EOF | oc apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: pvc-inspect
spec:
  containers:
  - name: shell
    image: quay.io/rcardona/walmart-train-demo:latest
    volumeMounts:
    - name: model-storage
      mountPath: /workspace
  volumes:
  - name: model-storage
    persistentVolumeClaim:
      claimName: model-storage
  restartPolicy: Never
EOF