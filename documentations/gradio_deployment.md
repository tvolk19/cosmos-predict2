
## Docker Deployment with Gradio Server

You can run Cosmos-Predict2 in a Docker container with a web-based Gradio interface for easy interaction.

### Building and Running the Container

Our [setup guide](setup.md) provides complete information on how to build a Docker container.

1. **Run the Docker container:**
   ```bash
   # Run the container with GPU support, port mapping, and volume mounts
   docker run --gpus all -it --rm \
     -p 8080:8080 \
     -v /path/to/cosmos-predict2:/workspace \
     -v /path/to/checkpoints:/workspace/checkpoints \
     -v /path/to/datasets:/workspace/datasets \
     <your docker>
   ```
2. **Inside the container, setup Gradio:**
   ```bash
   sed -i -e 's/h11==0.16.0/h11==0.14.0/g' /etc/pip/constraint.txt # remove problematic dependency for gradio
   python3 -m pip install gradio --break-system-packages
   ```

3. **Inside the container, launch the Gradio server:**
   ```bash
   # Set up the environment
   export PYTHONPATH=/workspace
   cd /workspace
   export NUM_GPU=1
   export FACTORY_MODULE="deployment.server.model_specific.video2world_worker"
   export MODEL_NAME="video2world"
   export CHECKPOINT_DIR="/workspace/checkpoints"

   # Launch the Gradio server
   python deployment/gradio_bootstrapper.py
   ```

4. **Access the web interface:**
   Open your browser and navigate to `http://localhost:8080`

### Gradio Server Configuration

The `gradio_bootstrapper.py` script automatically configures the Gradio interface based on your model setup:

- **Server Configuration**: Located in `deployment/server/generic/server_config.py`
- **Model Configuration**: Located in `deployment/server/model_specific/model_config.py`
- **Web Interface**: Automatically served on `0.0.0.0:8080`

### Supported Models

The Gradio interface will run model workers specified in the FACTORY_MODULE environment variable:

- `deployment.server.model_specific.text2image_worker` for **Text2Image**
- `deployment.server.model_specific.video2world_worker` for **Video2World**

The UI interface is configured based on the MODEL_NAME environment variable:
- `video2world`
- `text2image`
