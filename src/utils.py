import matplotlib.pyplot as plt
def visualize_sample(dataset, index):
    # Retrieve the sample
    image, target = dataset[index]

    # Display the image
    plt.imshow(image.permute(1, 2, 0))  # Convert from CHW to HWC
    plt.axis('off')

    # Draw bounding boxes
    boxes = target["boxes"]
    for box in boxes:
        x_center, y_center, width, height = box
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        rect = plt.Rectangle(
            (x_min, y_min), width, height,
            fill=False, color='red', linewidth=2
        )
        plt.gca().add_patch(rect)

    plt.show()
