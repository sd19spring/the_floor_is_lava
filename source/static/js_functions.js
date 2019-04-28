let dict = {};  // for storing whether the camera is turned on or off; needs to be globally accessible

function createDivs(numCaps) {
    // Dynamically creates the html elements used to display the camera feeds - numCaps defines how many cameras there
    // are to display.
    let i;  // for iteration
    for (i = numCaps; i >= 0; i--) {
        // Create a new, plain <span> element
        var frame = document.createElement("div");
        var eye_str = '/' + i;
        frame.setAttribute('class', 'capture_div');
        frame.innerHTML = "<img alt='Camera frame' src=" + eye_str + " class='capture'/><div onclick='muteCam(" + i + ")' id='capToggle" + i + "' class='cover button half' "
            + ">Mute</div>";
        // Get a reference to the element, before we want to insert the element
        var sp2 = document.getElementById("childElement");
        // Get a reference to the parent element
        var parentDiv = sp2.parentNode;
        // Insert the new element into the DOM before sp2
        parentDiv.insertBefore(frame, sp2.nextSibling);

        dict[i] = 1;  // indicate that the camera should be turned on
    }
}

// needs to be a global variable
let switchBool = '';
function changeSwitch(url) {
    // ajax the JSON to the server
    if (switchBool === 'true') {
        switchBool = 'false';
    } else {
        switchBool = 'true';
    }

    $.post(url, switchBool, changeButtonRecording(switchBool));
    // stop link reloading the page
     event.preventDefault();
}

function muteCam(capNum) {
    if (dict[capNum] === 1) {
        // turn the camera off
        $.post("/cap_switch", {capNum: capNum, record: 0}, changeButtonCam(capNum, dict[capNum]));
        // stop link reloading the page
        event.preventDefault();
        dict[capNum] = 0;
    } else {
        // turn the camera on
        $.post("/cap_switch", {capNum: capNum, record: 1}, changeButtonCam(capNum, dict[capNum]));
        // stop link reloading the page
        event.preventDefault();
        dict[capNum] = 1;
    }
}

function changeButtonRecording(switchBool) {
    if (switchBool === 'true') {
        document.getElementById("switch").innerHTML = "<a>Stop Recording</a>";
        document.getElementById("switch").className = "cover button_red";
    } else {
        document.getElementById("switch").innerHTML = "<a>Create Heatmap ➡️</a>";
        document.getElementById("switch").className = "cover button";
    }
}

function changeButtonCam(capNum, switchBool) {
    var elementID = '';
    elementID = 'capToggle' + capNum;
    if (switchBool === 1) {
        document.getElementById(elementID).innerHTML = "<a>Unmute</a>";
        document.getElementById(elementID).className = "cover button_red half";
    } else {
        document.getElementById(elementID).innerHTML = "<a>Mute️</a>";
        document.getElementById(elementID).className = "cover button half";
    }
}