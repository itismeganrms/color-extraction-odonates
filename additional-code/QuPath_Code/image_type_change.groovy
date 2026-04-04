// Get the current image
def imageData = getCurrentImageData()

// Set image type to OTHER
imageData.setImageType(ImageData.ImageType.OTHER)

// Update the display
fireHierarchyUpdate()
println("Image type set to OTHER.")
