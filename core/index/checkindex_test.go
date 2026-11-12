package index_test

import (
	"context"
	"encoding/json"
	"errors"
	"sort"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/geange/lucene-go/codecs/simpletext"
	"github.com/geange/lucene-go/core/document"
	coreindex "github.com/geange/lucene-go/core/index"
	idx "github.com/geange/lucene-go/core/interface/index"
	"github.com/geange/lucene-go/core/search"
	"github.com/geange/lucene-go/core/store"
)

// Check names are part of the serialized report contract.
const (
	checkReaderAccess = "readerAccess"
	checkLiveDocs     = "liveDocs"
	checkSegmentID    = "segmentId"
)

const testSoftDeletesField = "__softDelete"

// newTestIndexWriter builds a writer over dir using the SimpleText codec.
// When softDeleteField is non empty, soft deletes are enabled for it.
func newTestIndexWriter(t *testing.T, dir store.Directory, softDeleteField string) *coreindex.IndexWriter {
	t.Helper()

	codec := simpletext.NewCodec()
	similarity, err := search.NewBM25Similarity()
	require.NoError(t, err)

	conf := coreindex.NewIndexWriterConfig(codec, similarity)
	if softDeleteField != "" {
		conf.SetSoftDeletesField(softDeleteField)
	}

	writer, err := coreindex.NewIndexWriter(context.Background(), dir, conf)
	require.NoError(t, err)
	return writer
}

func newIDDoc(id string) *document.Document {
	doc := document.NewDocument()
	doc.Add(document.NewTextField("id", id, true))
	doc.Add(document.NewTextField("body", "body-of-"+id, false))
	return doc
}

func commitAndClose(t *testing.T, writer *coreindex.IndexWriter) {
	t.Helper()
	require.NoError(t, writer.Commit(context.Background()))
	require.NoError(t, writer.Close())
}

// leafNameComparator orders leaves by their committed segment name, which
// keeps multi-segment readers opened deterministically in tests.
func leafNameComparator(a, b idx.LeafReader) int {
	an := a.(*coreindex.SegmentReader).GetSegmentName()
	bn := b.(*coreindex.SegmentReader).GetSegmentName()
	return strings.Compare(an, bn)
}

// faultDirectory fails every OpenInput call that targets a data file of one
// specific segment. The segment info (.si) file stays readable so the commit
// itself can still be loaded, exercising per-segment failure isolation.
type faultDirectory struct {
	*store.RAMDirectory
	faultSegment string
}

func (d *faultDirectory) OpenInput(ctx context.Context, name string) (store.IndexInput, error) {
	if d.isFaultFile(name) {
		return nil, errors.New("simulated unreadable segment data")
	}
	return d.RAMDirectory.OpenInput(ctx, name)
}

func (d *faultDirectory) isFaultFile(name string) bool {
	if strings.HasSuffix(name, ".si") {
		return false
	}
	return name == d.faultSegment ||
		strings.HasPrefix(name, d.faultSegment+".") ||
		strings.HasPrefix(name, d.faultSegment+"_")
}

func sortedFiles(t *testing.T, dir store.Directory) []string {
	t.Helper()
	files, err := dir.ListAll(context.Background())
	require.NoError(t, err)
	sort.Strings(files)
	return files
}

func TestCheckIndex_EmptyIndex(t *testing.T) {
	dir := store.NewRAMDirectory()
	writer := newTestIndexWriter(t, dir, "")
	commitAndClose(t, writer)

	checker := coreindex.NewCheckIndex(dir)
	report, err := checker.DoCheck(coreindex.NewCheckIndexOptions())
	require.NoError(t, err)
	require.NotNil(t, report)

	assert.True(t, report.Healthy)
	assert.Equal(t, int64(1), report.CommitGeneration)
	assert.Equal(t, 0, report.MaxDoc)
	assert.Equal(t, 0, report.NumDocs)
	assert.Equal(t, 0, report.TotalDeletedDocs)
	assert.Equal(t, 0, report.TotalSoftDeletedDocs)
	assert.Equal(t, 0, report.SegmentCount)
	assert.NotNil(t, report.Segments)
	assert.Len(t, report.Segments, 0)
	assert.Empty(t, report.Error)

	// Repeated checks over an unchanged directory must be identical.
	repeat, err := checker.DoCheck(coreindex.NewCheckIndexOptions())
	require.NoError(t, err)
	assert.JSONEq(t, string(mustMarshalJSON(t, report)), string(mustMarshalJSON(t, repeat)))
}

func TestCheckIndex_MultipleSegments(t *testing.T) {
	dir := store.NewRAMDirectory()
	writer := newTestIndexWriter(t, dir, "")

	ctx := context.Background()
	_, err := writer.AddDocument(ctx, newIDDoc("1"))
	require.NoError(t, err)
	_, err = writer.AddDocument(ctx, newIDDoc("2"))
	require.NoError(t, err)
	require.NoError(t, writer.Commit(ctx))

	_, err = writer.AddDocument(ctx, newIDDoc("3"))
	require.NoError(t, err)
	commitAndClose(t, writer)

	report, err := coreindex.NewCheckIndex(dir).DoCheck(coreindex.NewCheckIndexOptions())
	require.NoError(t, err)
	require.True(t, report.Healthy)
	require.Len(t, report.Segments, 2)

	assert.Equal(t, 2, report.SegmentCount)
	assert.Equal(t, 3, report.MaxDoc)
	assert.Equal(t, 3, report.NumDocs)
	assert.Equal(t, 0, report.TotalDeletedDocs)

	first := report.Segments[0]
	second := report.Segments[1]

	// Commit order with cumulative doc bases.
	assert.Equal(t, "_0", first.Name)
	assert.Equal(t, 0, first.DocBase)
	assert.Equal(t, 2, first.MaxDoc)
	assert.Equal(t, 2, first.LiveDocs)
	assert.Equal(t, 0, first.DeletedDocs)
	assert.Equal(t, coreindex.SegmentStatusOK, first.Status)

	assert.Equal(t, "_1", second.Name)
	assert.Equal(t, 2, second.DocBase)
	assert.Equal(t, 1, second.MaxDoc)
	assert.Equal(t, 1, second.LiveDocs)
	assert.Equal(t, 0, second.DeletedDocs)
	assert.Equal(t, coreindex.SegmentStatusOK, second.Status)

	for _, segment := range report.Segments {
		require.Len(t, segment.Checks, 3)
		for _, check := range segment.Checks {
			assert.True(t, check.Passed, "check %s of %s should pass", check.Name, segment.Name)
			assert.Empty(t, check.Message)
		}
		assert.Equal(t, []string{checkReaderAccess, checkLiveDocs, checkSegmentID},
			checkNames(segment))
	}

	repeat, err := coreindex.NewCheckIndex(dir).DoCheck(coreindex.NewCheckIndexOptions())
	require.NoError(t, err)
	assert.Equal(t, mustMarshalJSON(t, report), mustMarshalJSON(t, repeat))
}

func TestCheckIndex_DeletedDocuments(t *testing.T) {
	dir := store.NewRAMDirectory()
	writer := newTestIndexWriter(t, dir, "")
	ctx := context.Background()

	for _, id := range []string{"1", "2", "3"} {
		_, err := writer.AddDocument(ctx, newIDDoc(id))
		require.NoError(t, err)
	}
	require.NoError(t, writer.Commit(ctx))

	// Updating by term hard-deletes the matching document of the old segment
	// and adds a replacement document in a new segment.
	_, err := writer.UpdateDocument(ctx, coreindex.NewTerm("id", []byte("2")), newIDDoc("2-updated"))
	require.NoError(t, err)
	commitAndClose(t, writer)

	report, err := coreindex.NewCheckIndex(dir).DoCheck(coreindex.NewCheckIndexOptions())
	require.NoError(t, err)
	require.True(t, report.Healthy)
	require.Len(t, report.Segments, 2)

	assert.GreaterOrEqual(t, report.TotalDeletedDocs, 1)
	assert.Equal(t, report.MaxDoc-report.TotalDeletedDocs, report.NumDocs)

	var deletedSegment *coreindex.SegmentReport
	for _, segment := range report.Segments {
		assert.Equal(t, coreindex.SegmentStatusOK, segment.Status)
		assert.Equal(t, segment.MaxDoc-segment.DeletedDocs, segment.LiveDocs)
		if segment.DeletedDocs > 0 {
			deletedSegment = segment
		}
	}
	require.NotNil(t, deletedSegment)
	assert.GreaterOrEqual(t, deletedSegment.DeletedDocs, 1)

	liveDocsCheck := findCheck(t, deletedSegment, checkLiveDocs)
	assert.True(t, liveDocsCheck.Passed)
}

func TestCheckIndex_SoftDeletes(t *testing.T) {
	dir := store.NewRAMDirectory()
	writer := newTestIndexWriter(t, dir, testSoftDeletesField)
	ctx := context.Background()

	_, err := writer.AddDocument(ctx, newIDDoc("1"))
	require.NoError(t, err)
	require.NoError(t, writer.Commit(ctx))

	_, err = writer.SoftUpdateDocument(ctx,
		coreindex.NewTerm("id", []byte("1")),
		newIDDoc("1-v2"),
		document.NewNumericDocValuesField(testSoftDeletesField, 1))
	require.NoError(t, err)
	commitAndClose(t, writer)

	report, err := coreindex.NewCheckIndex(dir).DoCheck(coreindex.NewCheckIndexOptions())
	require.NoError(t, err)
	require.True(t, report.Healthy)

	assert.GreaterOrEqual(t, report.TotalSoftDeletedDocs, 1)
	// Soft-deleted documents remain live for hard deletion accounting.
	assert.Equal(t, report.MaxDoc-report.TotalDeletedDocs, report.NumDocs)

	for _, segment := range report.Segments {
		assert.Equal(t, coreindex.SegmentStatusOK, segment.Status)
		assert.GreaterOrEqual(t, segment.SoftDeletedDocs, 0)
	}
}

func TestCheckIndex_SingleSegmentFailure(t *testing.T) {
	rawDir := store.NewRAMDirectory()
	writer := newTestIndexWriter(t, rawDir, "")
	ctx := context.Background()

	_, err := writer.AddDocument(ctx, newIDDoc("1"))
	require.NoError(t, err)
	require.NoError(t, writer.Commit(ctx))
	_, err = writer.AddDocument(ctx, newIDDoc("2"))
	require.NoError(t, err)
	commitAndClose(t, writer)

	before := sortedFiles(t, rawDir)

	dir := &faultDirectory{RAMDirectory: rawDir, faultSegment: "_1"}
	report, err := coreindex.NewCheckIndex(dir).DoCheck(coreindex.NewCheckIndexOptions())
	require.NoError(t, err)
	require.NotNil(t, report)
	require.Len(t, report.Segments, 2)
	assert.False(t, report.Healthy)

	// Aggregate counts stay deterministic even with one unreadable segment.
	assert.Equal(t, 2, report.SegmentCount)
	assert.Equal(t, 2, report.MaxDoc)
	assert.Equal(t, 2, report.NumDocs)

	healthy := report.Segments[0]
	failed := report.Segments[1]
	assert.Equal(t, "_0", healthy.Name)
	assert.Equal(t, coreindex.SegmentStatusOK, healthy.Status)
	assert.Equal(t, "_1", failed.Name)
	assert.Equal(t, coreindex.SegmentStatusFailed, failed.Status)
	require.NotEmpty(t, failed.Error)
	// Only public evidence: segment name and check names, no paths or raw IO text.
	assert.NotContains(t, failed.Error, "/")
	assert.NotContains(t, failed.Error, "simulated unreadable segment data")

	accessCheck := findCheck(t, failed, checkReaderAccess)
	assert.False(t, accessCheck.Passed)
	assert.NotEmpty(t, accessCheck.Message)
	assert.NotContains(t, accessCheck.Message, "/")

	// The healthy segment was still fully inspected despite the failure.
	assert.Len(t, healthy.Checks, 3)
	for _, check := range healthy.Checks {
		assert.True(t, check.Passed)
	}

	// Read-only guarantee: no file was created, removed or renamed.
	after := sortedFiles(t, rawDir)
	assert.Equal(t, before, after)
}

func TestCheckIndex_CancelledContext(t *testing.T) {
	dir := store.NewRAMDirectory()
	writer := newTestIndexWriter(t, dir, "")
	ctx := context.Background()

	for _, id := range []string{"1", "2"} {
		_, err := writer.AddDocument(ctx, newIDDoc(id))
		require.NoError(t, err)
		require.NoError(t, writer.Commit(ctx))
	}
	_, err := writer.AddDocument(ctx, newIDDoc("3"))
	require.NoError(t, err)
	commitAndClose(t, writer)

	cancelledCtx, cancel := context.WithCancel(context.Background())
	cancel()

	report, err := coreindex.NewCheckIndex(dir).DoCheck(&coreindex.CheckIndexOptions{Context: cancelledCtx})
	require.ErrorIs(t, err, context.Canceled)
	require.NotNil(t, report)
	assert.False(t, report.Healthy)
	assert.Len(t, report.Segments, report.SegmentCount)
	assert.GreaterOrEqual(t, report.SegmentCount, 2)

	for _, segment := range report.Segments {
		assert.Equal(t, coreindex.SegmentStatusCancelled, segment.Status)
		assert.NotEmpty(t, segment.Error)
		assert.Empty(t, segment.Checks)
	}
}

func TestCheckIndex_FromReader(t *testing.T) {
	dir := store.NewRAMDirectory()
	writer := newTestIndexWriter(t, dir, "")
	ctx := context.Background()

	_, err := writer.AddDocument(ctx, newIDDoc("1"))
	require.NoError(t, err)
	require.NoError(t, writer.Commit(ctx))
	_, err = writer.AddDocument(ctx, newIDDoc("2"))
	require.NoError(t, err)
	commitAndClose(t, writer)

	reader, err := coreindex.OpenDirectoryReader(ctx, dir, nil, leafNameComparator)
	require.NoError(t, err)
	defer func() { _ = reader.Close() }()

	directoryReport, err := coreindex.NewCheckIndex(dir).DoCheck(coreindex.NewCheckIndexOptions())
	require.NoError(t, err)

	report, err := coreindex.NewCheckIndex(dir).DoCheck(&coreindex.CheckIndexOptions{
		Context: ctx,
		Reader:  reader,
	})
	require.NoError(t, err)
	require.True(t, report.Healthy)
	assert.Equal(t, directoryReport.CommitGeneration, report.CommitGeneration)
	assert.Equal(t, directoryReport.SegmentCount, report.SegmentCount)
	assert.Equal(t, directoryReport.MaxDoc, report.MaxDoc)
	assert.Equal(t, directoryReport.NumDocs, report.NumDocs)

	for i, segment := range report.Segments {
		assert.Equal(t, directoryReport.Segments[i].Name, segment.Name)
		assert.Equal(t, directoryReport.Segments[i].DocBase, segment.DocBase)
		assert.Equal(t, coreindex.SegmentStatusOK, segment.Status)
	}

	// The supplied reader must remain owned by the caller and usable afterwards.
	assert.Greater(t, reader.NumDocs(), 0)
	leaves, err := reader.Leaves()
	require.NoError(t, err)
	assert.Len(t, leaves, report.SegmentCount)
}

func TestCheckIndex_NoCommit(t *testing.T) {
	dir := store.NewRAMDirectory()
	report, err := coreindex.NewCheckIndex(dir).DoCheck(coreindex.NewCheckIndexOptions())
	require.Error(t, err)
	require.NotNil(t, report)
	assert.False(t, report.Healthy)
	assert.NotEmpty(t, report.Error)
	assert.NotContains(t, report.Error, "/")
	assert.Len(t, report.Segments, 0)
}

func TestCheckIndex_ReportSerialization(t *testing.T) {
	dir := store.NewRAMDirectory()
	writer := newTestIndexWriter(t, dir, "")
	ctx := context.Background()
	_, err := writer.AddDocument(ctx, newIDDoc("1"))
	require.NoError(t, err)
	require.NoError(t, writer.Commit(ctx))
	_, err = writer.AddDocument(ctx, newIDDoc("2"))
	require.NoError(t, err)
	commitAndClose(t, writer)

	report, err := coreindex.NewCheckIndex(dir).DoCheck(coreindex.NewCheckIndexOptions())
	require.NoError(t, err)

	raw := mustMarshalJSON(t, report)

	var decoded map[string]any
	require.NoError(t, json.Unmarshal(raw, &decoded))

	for _, key := range []string{
		"commitGeneration", "maxDoc", "numDocs", "totalDeletedDocs",
		"totalSoftDeletedDocs", "segmentCount", "healthy", "segments",
	} {
		_, ok := decoded[key]
		assert.True(t, ok, "missing top-level key %s", key)
	}

	segments := decoded["segments"].([]any)
	first := segments[0].(map[string]any)
	for _, key := range []string{
		"name", "docBase", "maxDoc", "liveDocs", "deletedDocs",
		"softDeletedDocs", "status", "checks",
	} {
		_, ok := first[key]
		assert.True(t, ok, "missing segment key %s", key)
	}
	assert.Equal(t, "OK", first["status"])

	checks := first["checks"].([]any)
	assert.Len(t, checks, 3)

	// Serialization is stable across repeated checks of the unchanged index.
	repeat, err := coreindex.NewCheckIndex(dir).DoCheck(coreindex.NewCheckIndexOptions())
	require.NoError(t, err)
	assert.Equal(t, string(raw), string(mustMarshalJSON(t, repeat)))
}

func TestTestLiveDocs_StaysCompatible(t *testing.T) {
	dir := store.NewRAMDirectory()
	writer := newTestIndexWriter(t, dir, "")
	ctx := context.Background()

	for _, id := range []string{"1", "2", "3"} {
		_, err := writer.AddDocument(ctx, newIDDoc(id))
		require.NoError(t, err)
	}
	require.NoError(t, writer.Commit(ctx))
	_, err := writer.UpdateDocument(ctx, coreindex.NewTerm("id", []byte("2")), newIDDoc("2-updated"))
	require.NoError(t, err)
	commitAndClose(t, writer)

	reader, err := coreindex.OpenDirectoryReader(ctx, dir, nil, leafNameComparator)
	require.NoError(t, err)
	defer func() { _ = reader.Close() }()

	leaves, err := reader.Leaves()
	require.NoError(t, err)
	require.NotEmpty(t, leaves)

	for _, leaf := range leaves {
		sr, ok := leaf.LeafReader().(*coreindex.SegmentReader)
		require.True(t, ok)
		assert.NoError(t, coreindex.TestLiveDocs(sr))
	}
}

func mustMarshalJSON(t *testing.T, value any) []byte {
	t.Helper()
	data, err := json.Marshal(value)
	require.NoError(t, err)
	return data
}

func checkNames(segment *coreindex.SegmentReport) []string {
	names := make([]string, 0, len(segment.Checks))
	for _, check := range segment.Checks {
		names = append(names, check.Name)
	}
	return names
}

func findCheck(t *testing.T, segment *coreindex.SegmentReport, name string) coreindex.SegmentCheckResult {
	t.Helper()
	for _, check := range segment.Checks {
		if check.Name == name {
			return check
		}
	}
	t.Fatalf("check %s not found in segment %s", name, segment.Name)
	return coreindex.SegmentCheckResult{}
}
