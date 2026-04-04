import qupath.lib.analysis.features.ObjectMeasurements
import qupath.lib.analysis.features.ObjectMeasurements.ShapeFeatures


// Loop through all images in the project
for (entry in getProject().getImageList()) {
    def imageData = entry.readImageData()
    def hierarchy = imageData.getHierarchy()

    def annotations = hierarchy.getAnnotationObjects()
    if (annotations.isEmpty()) {
        println "No annotations in " + entry.getImageName()
        continue
    }

    // Get pixel calibration from the image
    def cal = imageData.getServer().getPixelCalibration()

    // Compute all available shape features for the annotations
    ObjectMeasurements.addShapeMeasurements(
        annotations,
        cal,
        ShapeFeatures.values() // adds all shape features
    )

    println entry.getImageName() + ": computed shape features for " + annotations.size() + " annotations"
    
    entry.saveImageData(imageData)
}