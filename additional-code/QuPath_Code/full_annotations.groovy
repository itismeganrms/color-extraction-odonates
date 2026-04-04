def imageData = getCurrentImageData()

// Define output path (relative to project)
def outputDir = buildFilePath(PROJECT_BASE_DIR, 'export')
mkdirs(outputDir)
def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
def path = buildFilePath(outputDir, name + ".tif")
//
//// Define how much to downsample during export (may be required for large images)
//double downsample = 8

// Create an ImageServer where the pixels are derived from annotations
def labelServer = new LabeledImageServer.Builder(imageData)
//  .backgroundLabel(0, ColorTools.WHITE) // Specify background label (usually 0 or 255)
//  .downsample(downsample)    // Choose server resolution; this should match the resolution at which tiles are exported
  .addLabel('head', 1)      // Choose output labels (the order matters!)
  .addLabel('tail', 2)
  .addLabel('torso', 3)
  .addLabel('wings', 4)
  .multichannelOutput(false) // If true, each label refers to the channel of a multichannel binary image (required for multiclass probability)
  .build()

// Write the image
writeImage(labelServer, path)
//
///// geoJSON part
//import qupath.lib.objects.PathObject
//import qupath.lib.projects.Projects
//import qupath.lib.io.GsonTools
//
//// Get the current project
//def project = getProject()
//def outputDir = buildFilePath(PROJECT_BASE_DIR, 'export')
//// Loop through all image entries in the project
//for (entry in project.getImageList()) {
//    // Open each image
//    def imageData = entry.readImageData()
//
//    // Get the image name (without the extension)
//    def imageName = entry.getImageName().replaceFirst(/\.[^.]+/, "")
//
//    // Define the output path (same folder as the project, with .geojson extension)
//    def outputPath = buildFilePath(outputDir, imageName + ".geojson")
//
//    // Set the current image data for processing
//    imageData.getHierarchy().getSelectionModel().clearSelection()
//
//    // Get the list of annotations from the image
//    def annotations = imageData.getHierarchy().getAnnotationObjects()
//    
//
//    if (!annotations.isEmpty()) {
//        // Write annotations to GeoJSON
//        //exportAnnotationsAsGeoJson(outputPath, annotations)
//        exportObjectsToGeoJson(annotations, outputPath, "FEATURE_COLLECTION")
//        print("Exported annotations for: " + imageName + " to " + outputPath)
//    } else {
//        print("No annotations found for: " + imageName)
//    }
//}
//
//// Function to export annotations as GeoJSON
//def exportAnnotationsAsGeoJson(String outputPath, List<PathObject> annotations) {
//    def gson = GsonTools.getInstance(true)  // Get a Gson instance (true enables pretty printing)
//    
//    def file = new File(outputPath)
//    file.withWriter('UTF-8') { writer -> gson.toJson(annotations, writer)  // Convert the annotations to JSON and write to file
//    }
//}