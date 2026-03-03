using UnityEngine;
using System.IO;
using System.Threading.Tasks;

#if USE_GLTFAST
using GLTFast;
#endif

/// <summary>
/// Loads WorldGen-exported glTF/GLB scenes at runtime.
/// Requires the GLTFast package (com.unity.cloud.gltfast).
/// </summary>
public class SceneLoader : MonoBehaviour
{
    [Header("Scene Configuration")]
    [Tooltip("Path to the WorldGen output directory (relative to StreamingAssets)")]
    public string sceneDirectory = "worldgen_output";

    [Tooltip("Name of the scene manifest file")]
    public string manifestFileName = "manifest.json";

    [Header("Runtime Settings")]
    [Tooltip("Automatically load scene on Start")]
    public bool autoLoad = true;

    [Tooltip("Scale factor for the imported scene")]
    public float importScale = 1.0f;

    private GameObject _sceneRoot;

    async void Start()
    {
        if (autoLoad)
        {
            await LoadScene();
        }
    }

    /// <summary>
    /// Load the WorldGen scene from the configured directory.
    /// </summary>
    public async Task LoadScene()
    {
        string basePath = Path.Combine(Application.streamingAssetsPath, sceneDirectory);
        string manifestPath = Path.Combine(basePath, manifestFileName);

        if (!File.Exists(manifestPath))
        {
            Debug.LogError($"WorldGen manifest not found at: {manifestPath}");
            return;
        }

        // Parse manifest
        string manifestJson = File.ReadAllText(manifestPath);
        SceneManifest manifest = JsonUtility.FromJson<SceneManifest>(manifestJson);

        Debug.Log($"Loading WorldGen scene: {manifest.objects.Length} objects");

        // Create scene root
        if (_sceneRoot != null)
        {
            Destroy(_sceneRoot);
        }
        _sceneRoot = new GameObject("WorldGen_Scene");
        _sceneRoot.transform.localScale = Vector3.one * importScale;

#if USE_GLTFAST
        // Load main scene GLB
        if (!string.IsNullOrEmpty(manifest.scene_file))
        {
            string scenePath = Path.Combine(basePath, manifest.scene_file);
            if (File.Exists(scenePath))
            {
                var gltf = new GltfImport();
                bool success = await gltf.Load(scenePath);
                if (success)
                {
                    await gltf.InstantiateMainSceneAsync(_sceneRoot.transform);
                    Debug.Log("Main scene loaded successfully");
                }
                else
                {
                    Debug.LogError("Failed to load main scene GLB");
                }
            }
        }
#else
        Debug.LogWarning(
            "GLTFast not installed. Add com.unity.cloud.gltfast package " +
            "and define USE_GLTFAST to enable glTF loading."
        );
#endif

        // Setup NavMesh if navmesh file exists
        if (!string.IsNullOrEmpty(manifest.navmesh_file))
        {
            string navmeshPath = Path.Combine(basePath, manifest.navmesh_file);
            if (File.Exists(navmeshPath))
            {
                SetupNavMesh(navmeshPath);
            }
        }

        Debug.Log("WorldGen scene loading complete");
    }

    private void SetupNavMesh(string navmeshObjPath)
    {
        // NavMesh baking should be handled by NavMeshBaker component
        NavMeshBaker baker = _sceneRoot.GetComponent<NavMeshBaker>();
        if (baker == null)
        {
            baker = _sceneRoot.AddComponent<NavMeshBaker>();
        }
        baker.BakeFromGeometry();
    }

    /// <summary>
    /// Unload the current scene.
    /// </summary>
    public void UnloadScene()
    {
        if (_sceneRoot != null)
        {
            Destroy(_sceneRoot);
            _sceneRoot = null;
        }
    }
}

/// <summary>
/// JSON manifest structure matching WorldGen's output.
/// </summary>
[System.Serializable]
public class SceneManifest
{
    public string version;
    public string generator;
    public string scene_file;
    public string navmesh_file;
    public string ground_file;
    public SceneObject[] objects;
}

[System.Serializable]
public class SceneObject
{
    public string name;
    public string file;
    public string texture;
    public int vertex_count;
    public int face_count;
    public bool interactable;
}
