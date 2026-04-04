// Script to iterate through all project entries and reload each image
def project = QPEx.getProject()

for (entry in project.getImageList()) {
    def path = entry.getImageName()
    println "Refreshing image: $path"

    def imageData = QPEx.loadImage(entry)
    QPEx.closeImage()
}
