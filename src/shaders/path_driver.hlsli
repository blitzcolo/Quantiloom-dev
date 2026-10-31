#ifndef QUANTILOOM_PATH_DRIVER_HLSLI
#define QUANTILOOM_PATH_DRIVER_HLSLI

// ============================================================================
// The path driver: raygen shades the vertices of a path itself
// ============================================================================
// Included by raygen only. It walks the tree the recursion walked, depth
// first and in the same order, with the vertex library the closest-hit stage
// runs (ShadeVertexA / Advance / CompleteChild / FoldVertex) and the sky
// evaluator the miss stage runs (ShadeMiss), so every draw is made from the
// stream state it was always made from.
//
// A ray traced from here is record-only (QlTraceRecord): closest-hit writes
// where it landed, miss says that it missed, and the vertex is shaded here.
// Every child a vertex asks for through Advance is descended into the same
// way -- both environment-bounce lobes, the refracted or reflected ray, the
// hero-split ray and the three RGB dispersion rays: the parent's frame is
// pushed, the child is shaded, and the parent resumes once the child's answer
// is in. Only the fusion interface's child, which phase A traces itself
// (TraceFusionInterface), is still traced through the hit stage, whose
// ShadeVertex shades that subtree recursively.
//
// One frame per raygen-shaded ancestor of the vertex being shaded. A vertex
// pushes only to take a child, and every child Advance can ask for is gated on
// depth < MAX_PATH_DEPTH: the bounce in SampleBounce, transmission where
// ShadeVertexA enables it. So the vertex at index i has depth i and at most
// MAX_PATH_DEPTH frames are live. Were that ever to stop holding, a child
// asked for with the stack full is traced through the hit stage instead --
// which is what happened to every child before, and gives the same answer.
//
// What a frame keeps beside the VertexFrame:
//   v       the vertex's own state as phase A left it, which the fold writes
//           its outputs into;
//   medium  the fusion medium statics as phase A left them. A child's phase A
//           overwrites them. The recursion never noticed, because each
//           closest-hit invocation had its own and QlTraceRadiance saved and
//           restored them around its trace; here there is one set per path,
//           so they are restored before the parent resumes;
//   kind    the kind of the child in flight, which CompleteChild needs.
// ============================================================================

#include "shade_vertex.hlsli"
#include "shade_miss.hlsli"

struct PathFrame {
    VertexFrame         f;
    PathVertex          v;
    FusionMediumContext medium;
    uint                kind;
};

// Run a vertex on from where its frame stands, as ShadeVertex's loop does,
// until it asks for a child (true, with the request in req) or it has folded
// (false, with its answer in v). Only a full stack -- unreachable, see above --
// has it trace the child through the hit stage and fold it in on the spot.
bool QlRunVertex(inout VertexFrame f, inout PathVertex v, inout ChildRequest req,
                 bool canDescend) {
    [loop] while (Advance(f, v.depth, req)) {
        if (canDescend) {
            return true;
        }
        CompleteChild(f, req.kind, QlTraceRadiance(req));
    }
    FoldVertex(f, v);
    return false;
}

// The whole path of one primary ray: vin is what the recursion's primary
// payload carried in, and the result is what it carried out.
//
// A freshly shaded vertex and a resuming one run on through separate calls of
// QlRunVertex on purpose. One shared call (9970461) made the frame and the
// vertex values carried around the loop from two sources, and measured slower
// in every render with a larger raygen stack, so it was reverted.
PathVertex RunPath(RayDesc ray, uint rayFlags, PathVertex vin) {
    PathFrame stack[MAX_PATH_DEPTH];
    uint sp = 0u;
    bool pathDone = false;
    [loop] while (!pathDone) {
        ChildRequest req = (ChildRequest)0;
        bool descend = false;

        // The vertex at the end of `ray`: phase A, then on until it asks for a
        // child or is finished. The hit ray is the one the closest-hit stage
        // reconstructs from its intrinsics.
        HitRecord rec;
        if (QlTraceRecord(ray, rayFlags, vin, rec)) {
            RayDesc hitRay = ray;
            hitRay.TMax = rec.t;
            VertexFrame f = (VertexFrame)0;
            ShadeVertexA(rec, hitRay, vin, f, req);
            if ((f.ctl & QL_CTL_FINISHED) == 0u) {
                if (QlRunVertex(f, vin, req, sp < MAX_PATH_DEPTH)) {
                    stack[sp].f = f;
                    stack[sp].v = vin;
                    stack[sp].medium = SaveFusionMedium();
                    stack[sp].kind = req.kind;
                    ++sp;
                    descend = true;
                }
            }
        } else {
            ShadeMiss(vin, ray);
        }

        // Unless it asked for a child, vin is now a finished vertex. Fold it
        // into its parent, and the parent into its own, until one of them asks
        // for another child or the primary vertex is done.
        [loop] while (!descend && !pathDone) {
            if (sp == 0u) {
                pathDone = true;
            } else {
                const uint top = sp - 1u;
                RestoreFusionMedium(stack[top].medium);
                CompleteChild(stack[top].f, stack[top].kind, vin);
                if (QlRunVertex(stack[top].f, stack[top].v, req, true)) {
                    stack[top].kind = req.kind;
                    descend = true;
                } else {
                    vin = stack[top].v;
                    sp = top;
                }
            }
        }

        if (descend) {
            // Built exactly as QlTraceRadiance builds it, the medium included:
            // the statics hold the parent's at this point.
            ray = req.ray;
            rayFlags = req.rayFlags;
            vin = QlFromRay(QlChildPayload(req));
        }
    }
    return vin;
}

#endif // QUANTILOOM_PATH_DRIVER_HLSLI
