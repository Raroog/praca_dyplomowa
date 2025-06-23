from google.cloud import vision_v1


def sample_async_batch_annotate_images(input_image_uris, prefix, batch_size=100):
    """Perform async batch image annotation."""
    client = vision_v1.ImageAnnotatorClient()
    features = [
        {"type_": vision_v1.Feature.Type.TEXT_DETECTION},
    ]
    requests = []
    for input_image_uri in input_image_uris:
        source = {"image_uri": input_image_uri}
        image = {"source": source}
        requests.append({"image": image, "features": features})

    # Each requests element corresponds to a single image.  To annotate more
    # images, create a request element for each image and add it to
    # the array of requests
    output_uri = f"gs://text-detection/{prefix}"
    gcs_destination = {"uri": output_uri}

    # The max number of responses to output in each JSON file
    output_config = {"gcs_destination": gcs_destination, "batch_size": batch_size}

    # print(requests, output_config)

    operation = client.async_batch_annotate_images(
        requests=requests, output_config=output_config
    )

    print("Waiting for operation to complete...")
    response = operation.result(timeout=None)

    # # The output is written to GCS with the provided output_uri as prefix
    gcs_output_uri = response.output_config.gcs_destination.uri
    print(f"Output written to GCS with prefix: {gcs_output_uri}")
