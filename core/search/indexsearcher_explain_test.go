package search

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/geange/lucene-go/codecs/simpletext"
	"github.com/geange/lucene-go/core/document"
	coreindex "github.com/geange/lucene-go/core/index"
	"github.com/geange/lucene-go/core/interface/index"
	"github.com/geange/lucene-go/core/store"
	"github.com/geange/lucene-go/core/types"
	"github.com/geange/lucene-go/core/util"
)

// explainTestBits is a util.Bits implementation that allows tests to mark documents as deleted.
type explainTestBits struct {
	live []bool
}

func (b *explainTestBits) Test(i uint) bool { return b.live[i] }
func (b *explainTestBits) Len() uint        { return uint(len(b.live)) }
func (b *explainTestBits) Words() []uint64  { return nil }

var _ util.Bits = &explainTestBits{}

// newExplainTestWriter creates a committed IndexWriter over a fresh temp directory.
func newExplainTestWriter(t *testing.T, dir store.Directory) *coreindex.IndexWriter {
	t.Helper()
	codec := simpletext.NewCodec()
	similarity, err := NewBM25Similarity()
	require.NoError(t, err)
	conf := coreindex.NewIndexWriterConfig(codec, similarity)
	writer, err := coreindex.NewIndexWriter(context.Background(), dir, conf)
	require.NoError(t, err)
	return writer
}

// addBodyDoc indexes a single TextField document.
func addBodyDoc(t *testing.T, writer *coreindex.IndexWriter, body string) {
	t.Helper()
	doc := document.NewDocument()
	doc.Add(document.NewTextField("body", body, true))
	_, err := writer.AddDocument(context.Background(), doc)
	require.NoError(t, err)
}

// newSingleSegmentSearcher builds a one-segment index from docs and opens an IndexSearcher.
func newSingleSegmentSearcher(t *testing.T, docs []string) (index.IndexSearcher, index.IndexReader) {
	t.Helper()
	dir, err := store.NewNIOFSDirectory(t.TempDir())
	require.NoError(t, err)

	writer := newExplainTestWriter(t, dir)
	for _, doc := range docs {
		addBodyDoc(t, writer, doc)
	}
	require.NoError(t, writer.Commit(context.Background()))
	require.NoError(t, writer.Close())

	reader, err := coreindex.OpenDirectoryReader(context.Background(), dir, nil, nil)
	require.NoError(t, err)

	searcher, err := NewIndexSearcher(reader)
	require.NoError(t, err)
	return searcher, reader
}

// newMultiSegmentSearcher builds one segment per writer session and opens an IndexSearcher
// whose leaves keep the session order.
func newMultiSegmentSearcher(t *testing.T, segments ...[]string) (index.IndexSearcher, index.IndexReader) {
	t.Helper()
	dir, err := store.NewNIOFSDirectory(t.TempDir())
	require.NoError(t, err)

	for _, docs := range segments {
		writer := newExplainTestWriter(t, dir)
		for _, doc := range docs {
			addBodyDoc(t, writer, doc)
		}
		require.NoError(t, writer.Commit(context.Background()))
		require.NoError(t, writer.Close())
	}

	reader, err := coreindex.OpenDirectoryReader(context.Background(), dir, nil, nil)
	require.NoError(t, err)

	searcher, err := NewIndexSearcher(reader)
	require.NoError(t, err)
	return searcher, reader
}

// searcherWithDeletedDoc wraps the single leaf reader with live docs that mark deletedLocalDoc
// as deleted, and returns an IndexSearcher on top of the wrapped reader.
func searcherWithDeletedDoc(t *testing.T, reader index.IndexReader, deletedLocalDoc int) index.IndexSearcher {
	t.Helper()
	leaves, err := reader.Leaves()
	require.NoError(t, err)
	require.Len(t, leaves, 1)

	segReader, ok := leaves[0].LeafReader().(*coreindex.SegmentReader)
	require.True(t, ok)

	maxDoc := segReader.MaxDoc()
	live := make([]bool, maxDoc)
	for i := range live {
		live[i] = true
	}
	live[deletedLocalDoc] = false

	wrapped, err := segReader.New(segReader.GetOriginalSegmentInfo(),
		&explainTestBits{live: live}, nil, maxDoc-1, true)
	require.NoError(t, err)

	searcher, err := NewIndexSearcher(wrapped)
	require.NoError(t, err)
	return searcher
}

func bodyTerm(value string) *TermQuery {
	return NewTermQuery(coreindex.NewTerm("body", []byte(value)))
}

func explainFloat(t *testing.T, e types.Explanation) float64 {
	t.Helper()
	v, ok := e.GetValue().(float64)
	require.True(t, ok, "explanation value is not float64: %T", e.GetValue())
	return v
}

// findExplanation walks the explanation tree depth-first and returns the first node whose
// description contains substr, or nil when no such node exists.
func findExplanation(e types.Explanation, substr string) types.Explanation {
	if strings.Contains(e.GetDescription(), substr) {
		return e
	}
	for _, detail := range e.GetDetails() {
		if found := findExplanation(detail, substr); found != nil {
			return found
		}
	}
	return nil
}

// topNScore returns the score of global docID as reported by SearchTopN, or -1 when absent.
func topNScore(t *testing.T, searcher index.IndexSearcher, q index.Query, docID int) (float64, bool) {
	t.Helper()
	top, err := searcher.SearchTopN(context.Background(), q, 20)
	require.NoError(t, err)
	for _, scoreDoc := range top.GetScoreDocs() {
		if scoreDoc.GetDoc() == docID {
			return scoreDoc.GetScore(), true
		}
	}
	return -1, false
}

func TestIndexSearcherExplainTermQuery(t *testing.T) {
	searcher, _ := newSingleSegmentSearcher(t, []string{
		"alpha beta gamma",
		"alpha beta",
		"gamma delta",
	})

	query := bodyTerm("alpha")
	queryBefore := query.String("")

	// a matching document: root node carries the final score and a readable query description
	expl, err := searcher.Explain(context.Background(), query, 0)
	require.NoError(t, err)
	require.True(t, expl.IsMatch())
	assert.Contains(t, expl.GetDescription(), "body:alpha")

	// the root score must be identical to the score produced by the normal search path
	expected, found := topNScore(t, searcher, query, 0)
	require.True(t, found)
	assert.InDelta(t, expected, explainFloat(t, expl), 1e-12)

	// child nodes describe term frequency, norm (field length), idf and boost statistics
	freqNode := findExplanation(expl, "freq, occurrences of term within document")
	require.NotNil(t, freqNode)
	assert.True(t, freqNode.IsMatch())

	require.NotNil(t, findExplanation(expl, "idf, computed as"))
	require.NotNil(t, findExplanation(expl, "fieldLength"))
	require.NotNil(t, findExplanation(expl, "avgFieldLength"))
	require.NotNil(t, findExplanation(expl, "boost"))
	require.NotNil(t, findExplanation(expl, "tfNorm"))
	require.NotNil(t, findExplanation(expl, "score(freq="))

	// a document that does not contain the term returns a no-match explanation
	missExpl, err := searcher.Explain(context.Background(), query, 2)
	require.NoError(t, err)
	assert.False(t, missExpl.IsMatch())
	require.NotNil(t, findExplanation(missExpl, "no matching term"))

	// Explain must not alter the query object
	assert.Equal(t, queryBefore, query.String(""))
}

func TestIndexSearcherExplainMatchAllDocsQuery(t *testing.T) {
	searcher, _ := newSingleSegmentSearcher(t, []string{
		"alpha beta gamma",
		"alpha beta",
	})

	query := NewMatchAllDocsQuery()
	expl, err := searcher.Explain(context.Background(), query, 1)
	require.NoError(t, err)
	require.True(t, expl.IsMatch())
	assert.InDelta(t, 1.0, explainFloat(t, expl), 1e-12)
	assert.Equal(t, "*:*", expl.GetDescription())

	// a MatchAllDocsQuery still cannot match a deleted document
	deletedSearcher := searcherWithDeletedDoc(t, searcher.GetIndexReader(), 0)
	deletedExpl, err := deletedSearcher.Explain(context.Background(), query, 0)
	require.NoError(t, err)
	assert.False(t, deletedExpl.IsMatch())
	assert.Contains(t, deletedExpl.GetDescription(), "deleted")
}

func TestIndexSearcherExplainBooleanQuery(t *testing.T) {
	searcher, _ := newSingleSegmentSearcher(t, []string{
		"alpha beta gamma", // doc 0
		"alpha beta",       // doc 1
		"gamma delta",      // doc 2
	})

	// MUST + SHOULD: doc 0 matches both, score is the sum of the two clause scores
	mustShould, err := NewBooleanQueryBuilder().
		AddQuery(bodyTerm("alpha"), index.OccurMust).
		AddQuery(bodyTerm("gamma"), index.OccurShould).
		Build()
	require.NoError(t, err)

	expl, err := searcher.Explain(context.Background(), mustShould, 0)
	require.NoError(t, err)
	require.True(t, expl.IsMatch())
	expected, found := topNScore(t, searcher, mustShould, 0)
	require.True(t, found)
	assert.InDelta(t, expected, explainFloat(t, expl), 1e-12)
	require.NotNil(t, findExplanation(expl, "sum of:"))

	// doc 2 fails the required alpha clause
	requiredMiss, err := searcher.Explain(context.Background(), mustShould, 2)
	require.NoError(t, err)
	assert.False(t, requiredMiss.IsMatch())
	require.NotNil(t, findExplanation(requiredMiss, "Failure"))

	// MUST_NOT: a matching prohibited clause makes the whole query fail
	prohibited, err := NewBooleanQueryBuilder().
		AddQuery(bodyTerm("alpha"), index.OccurMust).
		AddQuery(bodyTerm("beta"), index.OccurMustNot).
		Build()
	require.NoError(t, err)
	prohibitedExpl, err := searcher.Explain(context.Background(), prohibited, 0)
	require.NoError(t, err)
	assert.False(t, prohibitedExpl.IsMatch())
	require.NotNil(t, findExplanation(prohibitedExpl, "embedded prohibited clause"))

	// FILTER: the filter clause matches but contributes no score
	filtered, err := NewBooleanQueryBuilder().
		AddQuery(bodyTerm("alpha"), index.OccurFilter).
		AddQuery(bodyTerm("beta"), index.OccurShould).
		Build()
	require.NoError(t, err)
	filterExpl, err := searcher.Explain(context.Background(), filtered, 0)
	require.NoError(t, err)
	require.True(t, filterExpl.IsMatch())
	require.NotNil(t, findExplanation(filterExpl, "no scoring provided by filter clause"))
	betaScore, found := topNScore(t, searcher, bodyTerm("beta"), 0)
	require.True(t, found)
	assert.InDelta(t, betaScore, explainFloat(t, filterExpl), 1e-12)

	// minimumNumberShouldMatch: doc 0 satisfies 3 clauses, doc 2 only one
	minShould, err := NewBooleanQueryBuilder().
		SetMinimumNumberShouldMatch(2).
		AddQuery(bodyTerm("alpha"), index.OccurShould).
		AddQuery(bodyTerm("beta"), index.OccurShould).
		AddQuery(bodyTerm("gamma"), index.OccurShould).
		Build()
	require.NoError(t, err)

	minMatch, err := searcher.Explain(context.Background(), minShould, 0)
	require.NoError(t, err)
	assert.True(t, minMatch.IsMatch())

	minMiss, err := searcher.Explain(context.Background(), minShould, 2)
	require.NoError(t, err)
	assert.False(t, minMiss.IsMatch())
	require.NotNil(t, findExplanation(minMiss, "Failure to meet minimum number of optional clauses"))

	// none of the explanations may come from a panic
	assert.NotPanics(t, func() {
		_, e := searcher.Explain(context.Background(), prohibited, 1)
		require.NoError(t, e)
	})
}

func TestIndexSearcherExplainNoMatchTerm(t *testing.T) {
	searcher, _ := newSingleSegmentSearcher(t, []string{
		"alpha beta gamma",
		"alpha beta",
	})

	// term that does not exist anywhere in the index
	expl, err := searcher.Explain(context.Background(), bodyTerm("missing"), 0)
	require.NoError(t, err)
	assert.False(t, expl.IsMatch())
	assert.Contains(t, expl.GetDescription(), "doesn't match doc")
}

func TestIndexSearcherExplainDeletedDocument(t *testing.T) {
	searcher, reader := newSingleSegmentSearcher(t, []string{
		"alpha beta gamma", // doc 0
		"alpha beta",       // doc 1, will be marked deleted
		"gamma delta",      // doc 2
	})

	// sanity check before deletion: doc 1 matches alpha
	before, err := searcher.Explain(context.Background(), bodyTerm("alpha"), 1)
	require.NoError(t, err)
	require.True(t, before.IsMatch())

	deletedSearcher := searcherWithDeletedDoc(t, reader, 1)

	expl, err := deletedSearcher.Explain(context.Background(), bodyTerm("alpha"), 1)
	require.NoError(t, err)
	assert.False(t, expl.IsMatch())
	assert.Contains(t, expl.GetDescription(), "deleted")

	// the deleted document must also disappear from normal search
	top, err := deletedSearcher.SearchTopN(context.Background(), bodyTerm("alpha"), 10)
	require.NoError(t, err)
	for _, scoreDoc := range top.GetScoreDocs() {
		assert.NotEqual(t, 1, scoreDoc.GetDoc())
	}
}

func TestIndexSearcherExplainOutOfBounds(t *testing.T) {
	searcher, reader := newSingleSegmentSearcher(t, []string{
		"alpha beta gamma",
		"alpha beta",
	})
	maxDoc := reader.MaxDoc()

	_, err := searcher.Explain(context.Background(), bodyTerm("alpha"), -1)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "out of bounds")

	_, err = searcher.Explain(context.Background(), bodyTerm("alpha"), maxDoc)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "out of bounds")

	_, err = searcher.Explain(context.Background(), bodyTerm("alpha"), maxDoc+10)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "out of bounds")
}

func TestIndexSearcherExplainCancelledContext(t *testing.T) {
	searcher, _ := newSingleSegmentSearcher(t, []string{
		"alpha beta gamma",
		"alpha beta",
	})

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := searcher.Explain(ctx, bodyTerm("alpha"), 0)
	require.Error(t, err)
	assert.True(t, errors.Is(err, context.Canceled))
}

func TestIndexSearcherExplainSearchTopNConsistency(t *testing.T) {
	searcher, _ := newSingleSegmentSearcher(t, []string{
		"alpha beta gamma",
		"alpha beta",
		"gamma delta",
	})

	// TermQuery: every SearchTopN hit must carry the same score as its Explain root
	termQuery := bodyTerm("alpha")
	top, err := searcher.SearchTopN(context.Background(), termQuery, 10)
	require.NoError(t, err)
	require.NotEmpty(t, top.GetScoreDocs())
	for _, scoreDoc := range top.GetScoreDocs() {
		expl, err := searcher.Explain(context.Background(), termQuery, scoreDoc.GetDoc())
		require.NoError(t, err)
		require.True(t, expl.IsMatch())
		assert.InDelta(t, scoreDoc.GetScore(), explainFloat(t, expl), 1e-12)
	}

	// BooleanQuery with MUST clauses: explain and search must stay consistent
	mustQuery, err := NewBooleanQueryBuilder().
		AddQuery(bodyTerm("alpha"), index.OccurMust).
		AddQuery(bodyTerm("beta"), index.OccurMust).
		Build()
	require.NoError(t, err)

	mustTop, err := searcher.SearchTopN(context.Background(), mustQuery, 10)
	require.NoError(t, err)
	require.NotEmpty(t, mustTop.GetScoreDocs())
	for _, scoreDoc := range mustTop.GetScoreDocs() {
		expl, err := searcher.Explain(context.Background(), mustQuery, scoreDoc.GetDoc())
		require.NoError(t, err)
		require.True(t, expl.IsMatch())
		assert.InDelta(t, scoreDoc.GetScore(), explainFloat(t, expl), 1e-12)
	}
}

func TestIndexSearcherExplainMultiLeaf(t *testing.T) {
	// two segments, each one holding two documents -> global doc bases 0 and 2
	searcher, reader := newMultiSegmentSearcher(t,
		[]string{"alpha beta", "alpha beta"},
		[]string{"alpha beta", "alpha beta"},
	)
	leaves, err := reader.Leaves()
	require.NoError(t, err)
	require.Len(t, leaves, 2)
	assert.Equal(t, 0, leaves[0].DocBase())
	assert.Equal(t, 2, leaves[1].DocBase())

	query := bodyTerm("alpha")

	// global doc 3 lives in the second leaf (docBase 2, local doc 1)
	expl, err := searcher.Explain(context.Background(), query, 3)
	require.NoError(t, err)
	require.True(t, expl.IsMatch())

	expected, found := topNScore(t, searcher, query, 3)
	require.True(t, found)
	assert.InDelta(t, expected, explainFloat(t, expl), 1e-12)

	// every global doc resolves to its owning leaf without error
	for docID := 0; docID < reader.MaxDoc(); docID++ {
		e, err := searcher.Explain(context.Background(), query, docID)
		require.NoError(t, err)
		assert.True(t, e.IsMatch(), "doc %d should match", docID)
	}
}
