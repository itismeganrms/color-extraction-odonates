import qupath.lib.gui.commands.UpdateUrisCommand
import qupath.lib.projects.Projects

def project = Projects.getCurrentProject()

if (project == null) {
    println "No project loaded."
} else {
    UpdateUrisCommand.promptToUpdateUris(project)
}
