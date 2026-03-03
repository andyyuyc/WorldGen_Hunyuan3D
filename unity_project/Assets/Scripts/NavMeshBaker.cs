using UnityEngine;

#if UNITY_AI_NAVIGATION
using Unity.AI.Navigation;
#endif

/// <summary>
/// Bakes Unity NavMesh from imported WorldGen geometry.
/// Requires the AI Navigation package (com.unity.ai.navigation).
/// </summary>
public class NavMeshBaker : MonoBehaviour
{
    [Header("NavMesh Settings")]
    public float agentRadius = 0.6f;
    public float agentHeight = 2.0f;
    public float maxSlope = 45.0f;
    public float stepHeight = 0.9f;

    /// <summary>
    /// Bake NavMesh from child geometry.
    /// </summary>
    public void BakeFromGeometry()
    {
#if UNITY_AI_NAVIGATION
        // Find or add NavMeshSurface
        NavMeshSurface surface = GetComponent<NavMeshSurface>();
        if (surface == null)
        {
            surface = gameObject.AddComponent<NavMeshSurface>();
        }

        // Configure
        surface.collectObjects = CollectObjects.Children;
        surface.useGeometry = NavMeshCollectGeometry.RenderMeshes;

        // Set agent settings
        surface.agentTypeID = 0; // Default agent
        surface.overrideTileSize = true;
        surface.tileSize = 256;

        // Add static flags to all child renderers
        MeshRenderer[] renderers = GetComponentsInChildren<MeshRenderer>();
        foreach (var renderer in renderers)
        {
            renderer.gameObject.isStatic = true;

            // Add MeshCollider if not present (needed for NavMesh baking)
            if (renderer.GetComponent<MeshCollider>() == null)
            {
                MeshFilter mf = renderer.GetComponent<MeshFilter>();
                if (mf != null && mf.sharedMesh != null)
                {
                    MeshCollider mc = renderer.gameObject.AddComponent<MeshCollider>();
                    mc.sharedMesh = mf.sharedMesh;
                }
            }
        }

        // Bake
        surface.BuildNavMesh();
        Debug.Log($"NavMesh baked: {renderers.Length} renderers processed");
#else
        Debug.LogWarning(
            "AI Navigation package not installed. " +
            "Add com.unity.ai.navigation and define UNITY_AI_NAVIGATION."
        );
#endif
    }
}
